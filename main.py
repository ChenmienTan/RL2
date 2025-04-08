from typing import Dict, List, Optional
import argparse
import math
import importlib.util
import torch
from torch.utils.data import DistributedSampler, DataLoader
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForTokenClassification
)
from vllm import LLM, SamplingParams
from tqdm import tqdm
import wandb
from data import RLDataset
from utils import (
    get_auto_wrap_policy,
    dispatch,
    all_gather,
    initialize_global_process_group,
    offload_fsdp_model_to_cpu,
    load_fsdp_model_to_gpu,
    offload_fsdp_optimizer,
    load_fsdp_optimizer,
    get_seqlen_balanced_partitions
)
from models.ring_attn_utils import (
    substitute_hf_flash_attn,

)
from models import get_actor_model, get_critic_model

def actor_forward(
    model,
    minibatch: Dict[str, torch.Tensor],
    temperature: float,
    ring_attn_group: Optional[torch.Tensor] = None
) -> torch.Tensor:

    logits = model(
        input_ids=minibatch["states"],
        position_ids=minibatch["position_ids"],
        ring_attn_group=ring_attn_group,
        use_cache=False
    ).logits / temperature
    
    return torch.gather(
        logits.log_softmax(-1),
        dim=-1,
        index=minibatch["actions"].unsqueeze(-1)
    ).squeeze(-1) * minibatch["action_mask"]

def critic_forward(model, minibatch: Dict[str, torch.Tensor], ring_attn_group: Optional[torch.Tensor] = None) -> torch.Tensor:
    return model(
        input_ids=minibatch["states"],
        position_ids=minibatch["position_ids"],
        use_cache=False,
        ring_attn_group=ring_attn_group
    ).logits.squeeze(-1) * minibatch["action_mask"]

def accumulate_to_eos(minibatch: Dict[str, torch.Tensor], key: str) -> torch.Tensor:

    end_indices = torch.where(minibatch["eos_mask"])[1]
    start_indices = torch.cat((
        torch.LongTensor([0]).to(torch.cuda.current_device()),
        end_indices[:-1] + 1
    ))

    result = torch.zeros_like(minibatch[key])
    for start_idx, end_idx in zip(start_indices, end_indices):
        result[0, end_idx] = minibatch[key][0, start_idx:end_idx + 1].sum()
    
    return result

def compute_kl_term(
    logps: torch.Tensor,
    ref_logps: torch.Tensor,
    kl_estimator: str
) -> torch.Tensor:

    logp_diffs = logps - ref_logps
    if kl_estimator == "k1":
        return logp_diffs
    elif kl_estimator == "k2":
        return logp_diffs.pow(2) / 2
    elif kl_estimator == "k3":
        return logp_diffs + torch.exp(- logp_diffs) - 1
    else:
        raise NotImplementedError

def group_minibatches_into_batches(
    minibatches: List[Dict[str, torch.Tensor]],
    updates: int
) -> List[List[Dict[str, torch.Tensor]]]:

    n_minibatches_per_update = len(minibatches) // updates
    return [
        minibatches[update * n_minibatches_per_update:(update + 1) * n_minibatches_per_update]
        for update in range(updates)
    ]


class Trainer:

    def __init__(self, args, device_mesh, rollout_device_mesh, sp_device_mesh):

        self.args = args
        self.device_mesh = device_mesh
        self.rollout_device_mesh = rollout_device_mesh
        self.sp_device_mesh = sp_device_mesh

        self.reward_fn = self.prepare_reward_fn()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.sampler, self.train_dataloader = self.prepare_sampler_dataloader(args.train_data_path, True)
        _, self.test_dataloader = self.prepare_sampler_dataloader(args.test_data_path, False)
        self.model, self.optimizer = self.prepare_model_optimizer(args.model_name, "actor", True)
        if args.kl_coef > 0:
            self.ref_model, _ = self.prepare_model_optimizer(args.model_name, "actor", False)
        if args.critic_model_name is not None:
            self.critic, self.critic_optimizer = self.prepare_model_optimizer(
                args.critic_model_name, "critic", True
            )
        self.llm = self.prepare_inference_engine()
        self.prepare_logger()
        # self.prepare_ring_attn()

    def prepare_reward_fn(self):
        
        spec = importlib.util.spec_from_file_location("custom_module", self.args.reward_fn_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.reward_fn

    def prepare_sampler_dataloader(self, data_path: str, train: bool):

        dataset = RLDataset(
            self.args,
            data_path,
            self.tokenizer,
            self.args.rollout_per_prompt if train else 1
            # if train, each prompt will be repeated for rollout_per_prompt times
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.rollout_device_mesh["dp"].size(),
            rank=self.rollout_device_mesh["dp"].get_local_rank(),
            # sharded inference engines share identical data
            shuffle=train,
            drop_last=True
        )
        dataloader = DataLoader(
            dataset,
            (self.args.batch_size if train else len(dataset)) // self.rollout_device_mesh["dp"].size(),
            # if test, pack all data in a single batch
            sampler=sampler,
            collate_fn=dataset.collate_fn          
        )

        return sampler, dataloader

    def prepare_model_optimizer(self, model_name: str, model_type: str, train: bool):
        
        if model_type == "actor":
            model_cls = AutoModelForCausalLM
            model_kwargs = {}
            model = get_actor_model(model_name, **model_kwargs)
        elif model_type == "critic":
            model_cls = AutoModelForTokenClassification
            model_kwargs = {"num_labels": 1}
            model = get_critic_model(model_name)
        else:
            raise NotImplementedError

        if train and not self.args.disable_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        auto_wrap_policy = get_auto_wrap_policy(model)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16
        ) if train else None

        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            device_mesh=self.device_mesh,
            device_id=torch.cuda.current_device()
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.args.lr if model_type == "actor" else self.args.critic_lr,
            weight_decay=self.args.weight_decay
        ) if train else None

        offload_fsdp_model_to_cpu(model)
        return model, optimizer

    def prepare_ring_attn(self):
        if self.sp_device_mesh["sp"].size() == 1:
            return
        
        self.ring_attn_rank = self.sp_device_mesh["sp"].get_local_rank()
        self.ring_attn_ranks = list(range(self.args.sp_size))

        substitute_hf_flash_attn(self.ring_attn_group)

    @property
    def ring_attn_group(self):
        if self.sp_device_mesh["sp"].size() == 1:
            return None
        return self.sp_device_mesh["sp"].get_group()

    def prepare_inference_engine(self):

        torch.cuda.manual_seed(self.rollout_device_mesh["dp"].get_local_rank())
        self.rng_state = torch.cuda.get_rng_state()
        # unify the randomness within each dp process, see `rollout`
        
        self.train_sampling_params = SamplingParams(
            temperature=self.args.train_temperature,
            max_tokens=self.args.max_response_length
        )

        self.test_sampling_params = SamplingParams(
            temperature=self.args.test_temperature,
            max_tokens=self.args.max_response_length
        )    

        return LLM(
            self.args.model_name,
            tensor_parallel_size=self.args.tp_size,
            distributed_executor_backend="external_launcher",
            # SPMD, see https://github.com/vllm-project/vllm/issues/11400
            gpu_memory_utilization=self.args.gpu_memory_utilization,
            enable_sleep_mode=True,
            seed=self.rollout_device_mesh["dp"].get_local_rank(),
            # to save memory for update, see `prepare_update`
        )

    def prepare_logger(self):

        if not self.args.disable_wandb and self.device_mesh.get_rank() == 0:
            wandb.init(
                project=self.args.project,
                name=self.args.experiment_name,
                config=vars(self.args)
            )

    def prepare_rollout(self):
        # offload params and optimizer to save memory for rollout
        # upload the latest params to the inference engine
        
        offload_fsdp_optimizer(self.optimizer)
        with FSDP.summon_full_params(self.model, offload_to_cpu=True):
            state_dict = self.model.state_dict()
        offload_fsdp_model_to_cpu(self.model) # offload params here, or params cannot be summoned
        torch.cuda.empty_cache() # or llm.wake_up() will OOM
        self.llm.wake_up() # upload inference engine to GPU
        model = self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
        model.load_weights(((name, param) for name, param in state_dict.items()))

    def prepare_update(self, batch: Dict[str, List]) -> List[Dict[str, torch.Tensor]]:
        # offload inference engine to save memory for update
        # for a higher throughput, pack data to eliminate padding

        self.llm.sleep()

        seq_len_list = [
            len(prompt_token_id) + len(response_token_id) - 1
            for prompt_token_id, response_token_id
            in zip(batch["prompt_token_ids"], batch["response_token_ids"])
        ]
        n_minibatches = math.ceil(
            sum(seq_len_list) / (self.args.max_length_per_device * self.sp_device_mesh["sp"].size())
        )
        # at least n_minibatches minibatches are needed
        
        device_mesh = self.sp_device_mesh["dp"]
        rank, world_size = device_mesh.get_local_rank(), device_mesh.size()
        # The number of minibatches on each process needs to be a 
        # multiple of `actor_update_per_rollout` and (possibly)
        # `critic_update_per_rollout` so that the minibatches can be 
        # evenly group into several batches, with each batch being 
        # used for an update.
        multiple_of_per_process = math.lcm(
            self.args.actor_update_per_rollout,
            self.args.critic_update_per_rollout
        ) if hasattr(self, "critic") else self.args.actor_update_per_rollout
        multiple_of = world_size * multiple_of_per_process
        if n_minibatches % multiple_of != 0:
            n_minibatches += (multiple_of - n_minibatches % multiple_of)
        # data are grouped into n_minibatches balanced minibatches
        partitions: List[List[int]] = get_seqlen_balanced_partitions(
            seq_len_list, k_partitions=n_minibatches, equal_size=False
        )

        n_minibatches_per_process = n_minibatches // world_size
        self.partitions = partitions[rank * n_minibatches_per_process:(rank + 1) * n_minibatches_per_process]
        # record the partition to prepare for `compute_advantages`

        minibatches: List[Dict[str, torch.Tensor]] = []
        for partition in self.partitions:

            states, actions, position_ids = [], [], []
            rewards, action_mask, eos_mask = [], [], []
            for p in partition:

                prompt_token_id = batch["prompt_token_ids"][p]
                response_token_id = batch["response_token_ids"][p]
                reward: float = batch["rewards"][p]

                states.extend(prompt_token_id + response_token_id[:-1])
                actions.extend((len(prompt_token_id) - 1) * [0] + response_token_id)
                position_ids.extend(list(range(len(prompt_token_id) + len(response_token_id) - 1)))

                rewards.extend((len(prompt_token_id) + len(response_token_id) - 2) * [0] + [reward])
                action_mask.extend((len(prompt_token_id) - 1) * [0] + len(response_token_id) * [1])
                eos_mask.extend((len(prompt_token_id) + len(response_token_id) - 2) * [0] + [1])
                
            minibatches.append({
                "states": torch.LongTensor([states]).to(torch.cuda.current_device()),
                "actions": torch.LongTensor([actions]).to(torch.cuda.current_device()),
                "position_ids": torch.LongTensor([position_ids]).to(torch.cuda.current_device()),
                "rewards": torch.FloatTensor([rewards]).to(torch.cuda.current_device()),
                "action_mask": torch.LongTensor([action_mask]).to(torch.cuda.current_device()),
                "eos_mask": torch.LongTensor([eos_mask]).to(torch.cuda.current_device())
            })

        return minibatches

    def rollout(self, batch: Dict[str, List], train: bool) -> Dict[str, List]:

        # unify the randomness of each dp process, or llm.generate will hang
        torch.cuda.set_rng_state(self.rng_state)
        responses = self.llm.generate(
            batch["prompts"],
            sampling_params=self.train_sampling_params if train else self.test_sampling_params,
            use_tqdm=(self.device_mesh.get_rank() == 0)
        )
        batch.update({
            "responses": [response.outputs[0].text for response in responses],
            "response_token_ids": [list(response.outputs[0].token_ids) for response in responses]
        })
        
        # Each device only rewards its respective data to avoid duplicate computation
        batch = dispatch(batch, self.rollout_device_mesh["tp"])
        # only support outcome reward, RM should be served remotely if there is
        batch["rewards"] = [
            self.reward_fn(response, answer)
            for response, answer in zip(batch["responses"], batch["answers"])
        ]

        batch = all_gather(batch) # will be dispatched in `prepare_update`
        if self.device_mesh.get_rank() == 0 and not self.args.disable_wandb:
            wandb.log({
                f"reward/{'train' if train else 'test'}": torch.Tensor(batch["rewards"]).mean().item(),
                f"response_length/{'train' if train else 'test'}": torch.Tensor([
                    len(response_token_id) for response_token_id in batch["response_token_ids"]
                ]).mean().item()
            }, step=self.step)

        return batch

    @torch.no_grad()
    def compute_log_probs(self, minibatches: List[Dict[str, torch.Tensor]], prefix: str):
        model = self.model if prefix == "old" else self.ref_model
        load_fsdp_model_to_gpu(model)

        for minibatch in (
            tqdm(minibatches, desc=f"Step {self.step + 1}, compute {prefix} log probs")
            if self.device_mesh.get_rank() == 0 else minibatches
        ):
            minibatch[f"{prefix}_logps"] = actor_forward(model, minibatch, self.args.train_temperature, self.ring_attn_group)
            
        offload_fsdp_model_to_cpu(model)

    @torch.no_grad()
    def compute_values(self, minibatches: List[Dict[str, torch.Tensor]]):
        load_fsdp_model_to_gpu(self.critic)

        self.critic.eval()
        for minibatch in (
            tqdm(minibatches, desc=f"Step {self.step + 1}, compute values")
            if self.device_mesh.get_rank() == 0 else minibatches
        ):
            minibatch["values"] = critic_forward(self.critic, minibatch, self.ring_attn_group)
        
        # no need to offload critic because it will be updated soon

    def compute_advantages(self, minibatches: List[Dict[str, torch.Tensor]]):

        # add KL term into rewards
        if self.args.kl_type == "reward":
            
            for minibatch in minibatches:

                if self.args.kl_level == "token":
                    logps, ref_logps = minibatch["old_logps"], minibatch["ref_logps"]
                elif self.args.kl_level == "sequence":
                    logps = accumulate_to_eos(minibatch, "old_logps")
                    ref_logps = accumulate_to_eos(minibatch, "ref_logps")
                else:
                    raise NotImplementedError

                # The logps and ref_logps of non-objective tokens are 
                # zero, so their kl_term will also be zero
                kl_term = compute_kl_term(logps, ref_logps, self.args.kl_estimator)
                minibatch["rewards"] -= self.args.kl_coef * kl_term

                # TODO: log KL

        if hasattr(self, "critic"):

            for minibatch in minibatches:

                # delta_t = r_t + \gamma * V(s_{t+1}) - V(s_t)
                # if s_{t+1} is a terminal state, V(s_{t+1}) = 0
                next_values = torch.cat(
                    (minibatch["values"][:, 1:], torch.FloatTensor([[0]]).to(torch.cuda.current_device())),
                dim=-1)
                delta = minibatch["rewards"] + self.args.gamma * (1 - minibatch["eos_mask"]) * next_values - minibatch["values"]

                # A_t = \delta_t + \gamma * \lambda * A_{t+1}
                # if s_{t+1} is a terminal state, A_{t+1} = 0
                gae, reversed_gaes = 0, []
                for t in reversed(range(delta.shape[-1])):
                    gae = delta[0, t] + self.args.gamma * self.args.lamda * (1 - minibatch["eos_mask"][0, t]) * gae
                    reversed_gaes.append(gae)
                gaes = reversed_gaes[::-1]

                minibatch["advantages"] = torch.FloatTensor([gaes]).to(torch.cuda.current_device()) * minibatch["action_mask"]
                minibatch["returns"] = minibatch["advantages"] + minibatch["values"]

        else:

            # recompute outcome rewards after adding the kl term
            rewards = torch.zeros((self.args.batch_size * self.args.rollout_per_prompt), device=torch.cuda.current_device())
            for minibatch, partition in zip(minibatches, self.partitions):
                # perhaps kl_type=reward, kl_level=token, then the 
                # rewards of non-eos tokens are non-zero
                reward = accumulate_to_eos(minibatch, "rewards")
                reward = reward[torch.where(minibatch["eos_mask"])]
                rewards[partition] = reward
            dist.all_reduce(rewards, op=dist.ReduceOp.SUM)
            
            # A(x,y_n) = R(x,y_n) - mean(R(x,y_{1:N}))
            rewards = rewards.view(self.args.batch_size, self.args.rollout_per_prompt)
            baselines = rewards.mean(-1, keepdim=True)
            advantages = rewards - baselines
            if self.args.group_norm:
                stds = rewards.std(-1, keepdim=True)
                advantages /= (stds + torch.finfo(stds.dtype).eps)
            advantages = advantages.flatten()

            # add advantages into the minibatches
            for minibatch, partition in zip(minibatches, self.partitions):
                end_indices = torch.where(minibatch["eos_mask"])[1]
                start_indices = torch.cat((
                    torch.LongTensor([0]).to(torch.cuda.current_device()),
                    end_indices[:-1] + 1
                ))
                minibatch["advantages"] = torch.zeros_like(minibatch["rewards"])
                for start_idx, end_idx, advantage in zip(start_indices, end_indices, advantages[partition]):
                    minibatch["advantages"][0, start_idx:end_idx + 1] = advantage
                minibatch["advantages"] *= minibatch["action_mask"]

    def update_critic(self, minibatches: List[Dict[str, torch.Tensor]]):
        # critic has been loaded on GPU in `compute_values`
        load_fsdp_optimizer(self.critic_optimizer, torch.cuda.current_device())

        self.critic.train()
        
        batches = group_minibatches_into_batches(
            minibatches, self.args.critic_update_per_rollout
        )
        losses, grad_norms = [], []
        for batch in (
            tqdm(batches, desc=f"Step {self.step + 1}, update critic")
            if self.device_mesh.get_rank() == 0 else batches
        ):

            total_actions = sum([minibatch["action_mask"].sum() for minibatch in batch])
            for minibatch in batch:

                values = critic_forward(self.critic, minibatch)
                cliped_values = torch.clamp(
                    values,
                    minibatch["values"] - self.args.value_clip,
                    minibatch["values"] + self.args.value_clip
                )
                loss = (torch.max(
                    (values - minibatch["returns"]).pow(2),
                    (cliped_values - minibatch["returns"]).pow(2)
                )).sum() / total_actions
                loss.backward()
                losses.append(loss.item())

            grad_norm = self.critic.clip_grad_norm_(self.args.max_grad_norm)
            grad_norms.append(grad_norm.item())
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()

        if not self.args.disable_wandb:
            losses = all_gather(losses)
            grad_norms = all_gather(grad_norms)
            if self.device_mesh.get_rank() == 0:
                wandb.log({
                    "critic/loss": torch.FloatTensor(losses).mean().item(),
                    "critic/grad_norm": torch.FloatTensor(grad_norms).mean().item()
                }, step=self.step)

        offload_fsdp_model_to_cpu(self.critic)
        offload_fsdp_optimizer(self.critic_optimizer)

    def update_actor(self, minibatches: List[Dict[str, torch.Tensor]]):
        load_fsdp_model_to_gpu(self.model)
        load_fsdp_optimizer(self.optimizer, torch.cuda.current_device())

        batches = group_minibatches_into_batches(
            minibatches, self.args.actor_update_per_rollout
        )
        losses, grad_norms = [], []
        for batch in (
            tqdm(batches, desc=f"Step {self.step + 1}, update actor")
            if self.device_mesh.get_rank() == 0 else batches
        ):
            total_actions = sum([minibatch["action_mask"].sum() for minibatch in batch])
            total_sequences = sum([(minibatch["eos_mask"]).sum() for minibatch in batch])
            for minibatch in batch:
                logps = actor_forward(self.model, minibatch, self.args.train_temperature, self.ring_attn_group)
                ratio = (logps - minibatch["old_logps"]).exp()
                cliped_ratio = torch.clamp(ratio, 1 - self.args.epsilon_clip, 1 + self.args.epsilon_clip)
                loss = - torch.min(minibatch["advantages"] * ratio, minibatch["advantages"] * cliped_ratio).sum() / total_actions

                # TODO: log cliped_ratio
                if self.args.kl_type == "loss":
                    
                    if self.args.kl_level == "token":
                        ref_logps = minibatch["ref_logps"]
                    elif self.args.kl_level == "sequence":
                        minibatch["logps"] = logps
                        logps = accumulate_to_eos(minibatch, "logps")
                        ref_logps = accumulate_to_eos(minibatch, "ref_logps")
                    else:
                        raise NotImplementedError

                    kl_term = compute_kl_term(
                        logps, ref_logps, self.args.kl_estimator
                    ) # TODO: log KL
                    kl_loss = - self.args.kl_coef * kl_term.sum() / (total_actions if self.args.kl_level == "token" else total_sequences)
                    loss = loss + kl_loss

                loss.backward()
                losses.append(loss.item())

            grad_norm = self.model.clip_grad_norm_(self.args.max_grad_norm)
            grad_norms.append(grad_norm.item())
            self.optimizer.step()
            self.optimizer.zero_grad()

        if not self.args.disable_wandb:
            losses = all_gather(losses)
            grad_norms = all_gather(grad_norms)
            if self.device_mesh.get_rank() == 0:
                wandb.log({
                    "actor/loss": torch.FloatTensor(losses).mean().item(),
                    "actor/grad_norm": torch.FloatTensor(grad_norms).mean().item()
                }, step=self.step)

    def save(self):

        load_fsdp_model_to_gpu(self.model) # or the params cannot be summoned
        with FSDP.summon_full_params(
            self.model,
            offload_to_cpu=True,
            rank0_only=True,
            writeback=False
        ):
            if self.device_mesh.get_rank() == 0:
                self.tokenizer.save_pretrained(f"{self.args.save_path}/{self.args.experiment_name}/step{self.args.step}")
                self.model.module.save_pretrained(f"{self.args.save_path}/{self.args.experiment_name}/step{self.args.step}")
        offload_fsdp_model_to_cpu(self.model)

    def train(self):

        self.model.train()

        self.step = 0
        for batch in self.test_dataloader:
            self.rollout(batch, False)
    
        for epoch in range(self.args.n_epochs):
            self.sampler.set_epoch(epoch)
            for batch in self.train_dataloader:

                batch = self.rollout(batch, True)
                minibatches = self.prepare_update(batch)

                self.compute_log_probs(minibatches, "old")
                if hasattr(self, "ref_model"):
                    self.compute_log_probs(minibatches, "ref")

                if hasattr(self, "critic"):
                    self.compute_values(minibatches)

                self.compute_advantages(minibatches)

                if hasattr(self, "critic"):
                    self.update_critic(minibatches)

                self.update_actor(minibatches)
                self.prepare_rollout()

                self.step += 1
                if self.step % self.args.test_freq == 0:
                    for batch in self.test_dataloader:
                        self.rollout(batch, False)
                    
                if self.step % self.args.save_freq == 0:
                    self.save()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str)
    parser.add_argument("--experiment_name", type=str)
    
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--disable_gradient_checkpointing", action="store_true")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument(
        "--sp_size", type=int, default=1,
        help="Sequence parallelism will be enabled if larger than one."
    )

    parser.add_argument(
        "--epsilon_clip", type=float, default=0.2,
        help="The clip value for actor in the PPO surrogant objective."
    )
    parser.add_argument(
        "--value_clip", type=float, default=0.5,
        help="The clip value for critic in PPO."
    )

    parser.add_argument(
        "--kl_coef", type=float, default=0.0,
        help="KL regularization (and therefore reference model) will be used if larger than zero."
    )
    # OpenAI PPO and OpenRLHF REINFORCE++ corresponds to kl_type=reward, kl_level=token, kl_estimator=k1
    # DeepSeek GRPO corresponds to kl_type=loss, kl_level=sequence, kl_estimator=k3
    parser.add_argument(
        "--kl_type", choices=["reward", "loss"], default=None,
        help="If `reward`, KL term will be added into the reward in `compute_advantages`. If `loss`, KL term will be added into the loss in `update_actor`."
    )
    parser.add_argument(
        "--kl_level", choices=["token", "sequence"], default=None,
        help="If `token` (resp. `sequence`), KL term of each `token` (resp. `sequence`) will be computed."
    )
    parser.add_argument(
        "--kl_estimator", choices=["k1", "k2", "k3"], default=None,
        help="The estimator of KL divergence. See http://joschu.net/blog/kl-approx.html."
    )

    parser.add_argument(
        "--group_norm", action="store_true",
        help="Devide the advantages by the group standard deviation. See https://arxiv.org/pdf/2402.03300."
    )

    parser.add_argument(
        "--critic_model_name", type=str, default=None,
        help="Generalized advantage estimator will be used if not None.  See https://arxiv.org/pdf/1506.02438."
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lamda", type=float, default=1.0)

    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--max_prompt_length", type=int)
    parser.add_argument("--max_response_length", type=int)

    parser.add_argument("--train_temperature", type=float, default=1.0)
    parser.add_argument("--test_temperature", type=float, default=0.0)

    parser.add_argument("--reward_fn_path", type=str)

    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument(
        "--batch_size", type=int,
        help="The number of prompts used in each rollout."
    )
    parser.add_argument(
        "--rollout_per_prompt", type=int,
        help="The number of responses sampled for each prompt."
    )
    parser.add_argument(
        "--actor_update_per_rollout", type=int, default=1,
        help="The number of actor updates corrsponds to a rollout."
    )
    parser.add_argument(
        "--critic_update_per_rollout", type=int, default=4,
        help="The number of critic updates corrsponds to a rollout."
    )
    parser.add_argument("--max_length_per_device", type=int)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--critic_lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--test_freq", type=int, default=-1)
    parser.add_argument("--save_freq", type=int, default=-1)
    parser.add_argument("--save_path", type=str, default="ckpts")
    args = parser.parse_args()

    _, _, world_size = initialize_global_process_group()
    # Use ZeRO stage 3 to shard params, grads, and optimizer states
    device_mesh = dist.device_mesh.init_device_mesh(
        "cuda",
        mesh_shape=(world_size,)
    )
    # Use tensor parallelism to shard params of inference engine
    rollout_device_mesh = dist.device_mesh.init_device_mesh(
        "cuda",
        mesh_dim_names=("dp", "tp"),
        mesh_shape=(world_size // args.tp_size, args.tp_size)
    )
    # Use ring attention to shard sequence activations
    sp_device_mesh = dist.device_mesh.init_device_mesh(
        "cuda",
        mesh_dim_names=("dp", "sp"),
        mesh_shape=(world_size // args.sp_size, args.sp_size)
    )

    trainer = Trainer(args, device_mesh, rollout_device_mesh, sp_device_mesh)
    trainer.train()

if __name__ == "__main__":
    main()

# TODO: support RingAttention