from typing import Dict, List
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
    compute_seq_logps,
    compute_kl_term,
    initialize_global_process_group,
    offload_fsdp_model_to_cpu,
    load_fsdp_model_to_gpu,
    offload_fsdp_optimizer,
    load_fsdp_optimizer,
    get_seqlen_balanced_partitions
)
from models.ring_attn_utils import (
    set_ring_attn_group,
    get_ring_attn_group
)

class Trainer:

    def __init__(self, args, device_mesh):

        self.args = args
        self.device_mesh = device_mesh

        self.rank = device_mesh.get_rank()
        self.world_size = device_mesh.size()

        self.rollout_global_world_size = self.world_size // self.args.tp_size
        self.rollout_global_rank = self.rank // self.args.tp_size
        self.rollout_local_rank = self.rank % self.args.tp_size
        
        self.device = torch.cuda.current_device()

        self.reward_fn = self.prepare_reward_fn()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.sampler, self.train_dataloader = self.prepare_sampler_dataloader(args.train_data_path, True)
        _, self.test_dataloader = self.prepare_sampler_dataloader(args.test_data_path, False)
        self.model, self.optimizer = self.prepare_model_optimizer(args.model_name, "actor", True)
        if args.critic_model_name is not None:
            self.critic, self.critic_optimizer = self.prepare_model_optimizer(
                args.critic_model_name, "critic", True
            )
        if args.kl_coef > 0:
            self.ref_model, _ = self.prepare_model_optimizer(args.model_name, "actor", False)
        self.prepare_ring_attn()
        self.llm = self.prepare_inference_engine()
        self.prepare_logger()

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
            shuffle=train,
            num_replicas=self.rollout_global_world_size,
            rank=self.rollout_global_rank
            # sharded inference engines share identical data
        )
        loader = DataLoader(
            dataset,
            (self.args.batch_size if train else len(dataset)) // self.rollout_global_world_size,
            # if test, pack all data in a single batch
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            drop_last=True            
        )

        return sampler, loader

    def prepare_model_optimizer(self, model_name, model_type, train):
        
        if model_type == "actor":
            model_cls = AutoModelForCausalLM
            model_kwargs = {}
        elif model_type == "critic":
            model_cls = AutoModelForTokenClassification
            model_kwargs = {"num_labels": 1}
        else:
            raise NotImplementedError

        model = model_cls.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if train else torch.bfloat16,
            attn_implementation="flash_attention_2",
            **model_kwargs
        )

        if train and not self.args.disable_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        auto_wrap_policy = get_auto_wrap_policy(model)

        if train:
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16
            )

        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision if train else None,
            device_mesh=self.device_mesh,
            device_id=self.device
        )

        if train:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        else:
            optimizer = None

        offload_fsdp_model_to_cpu(model)
        return model, optimizer

    def prepare_ring_attn(self):
        self.ring_attn_size = self.args.ring_attn_size
        if self.ring_attn_size == 1:
            self.ring_attn_rank = 0
            return
        
        ring_head_stride = getattr(self.args, "ring_head_stride", 1)
        for i in range(self.world_size // self.ring_attn_size):
            ring_attn_ranks = list(
                range(i * self.ring_attn_size, (i + 1) * self.ring_attn_size)
            )
            group = dist.new_group(ring_attn_ranks, backend="nccl")
            if self.rank in ring_attn_ranks:
                set_ring_attn_group(group)
                self.ring_attn_rank = dist.get_rank(group=group)
                self.ring_attn_ranks = ring_attn_ranks

        from ring_flash_attn import substitute_hf_flash_attn
        substitute_hf_flash_attn(self.ring_attn_group, ring_head_stride)

    @property
    def ring_attn_group(self):
        return get_ring_attn_group()

    def prepare_inference_engine(self):
        
        self.train_sampling_params = SamplingParams(
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
            # to save memory for update, see `prepare_update`
            seed=self.rollout_global_rank
        )

    def prepare_logger(self):

        if not self.args.disable_wandb and self.rank == 0:
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
        torch.cuda.empty_cache() # or self.llm.wake_up() will OOM
        self.llm.wake_up() # upload inference engine to GPU
        model = self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
        model.load_weights(((name, param) for name, param in state_dict.items()))

    def prepare_update(self, batch):
        # offload inference engine to save memory for update
        # pack data for a higher throughput

        self.llm.sleep()

        seq_len_list = [
            len(prompt_token_id) + len(response_token_id)
            for prompt_token_id, response_token_id
            in zip(batch["prompt_token_ids"], batch["response_token_ids"])
        ]
        batch_size = math.ceil(sum(seq_len_list) / self.args.max_length_per_device)
        # at least batch_size batches is needed
        if batch_size % self.world_size != 0:
            batch_size += (self.world_size - batch_size % self.world_size)
        # so that data can be evenly distributed to data processes
        batch_size_per_process = batch_size // self.world_size
        # data are grouped into batch_size balanced batches
        partitions: List[List[int]] = get_seqlen_balanced_partitions(
            seq_len_list, k_partitions=batch_size, equal_size=False
        )
        partitions = partitions[self.rank * batch_size_per_process:(self.rank + 1) * batch_size_per_process]

        batches = []
        for partition in partitions:

            states, actions, position_ids = [], [], []
            rewards, advantages = [], []
            action_mask, eos_mask = [], []
            for p in partition:

                prompt_token_id = batch["prompt_token_ids"][p]
                response_token_id = batch["response_token_ids"][p]
                reward = batch["rewards"][p]
                advantage = batch["advantages"][p]

                states.extend(prompt_token_id + response_token_id[:-1])
                actions.extend(prompt_token_id[1:] + response_token_id)
                position_ids.extend(list(range(len(prompt_token_id) + len(response_token_id) - 1)))

                rewards.extend((len(prompt_token_id) - 1) * [0] + reward)
                advantages.extend((len(prompt_token_id) - 1) * [0] + advantage)

                action_mask.extend((len(prompt_token_id) - 1) * [0] + len(response_token_id) * [1])
                eos_mask.extend((len(prompt_token_id) + len(response_token_id) - 2) * [1] + [0])
                
            batches.append({
                "states": torch.LongTensor([states]).to(self.device),
                "actions": torch.LongTensor([actions]).to(self.device),
                "position_ids": torch.LongTensor([position_ids]).to(self.device),
                "rewards": torch.FloatTensor([rewards]).to(self.device),
                "advantages": torch.FloatTensor([advantages]).to(self.device),
                "action_mask": torch.LongTensor([action_mask]).to(self.device),
                "eos_mask": torch.LongTensor([eos_mask]).to(self.device)
            })

        return batches

    def rollout(self, batch: Dict[str, List], train: bool):
        # sample responses and compute reward and advantages

        responses = self.llm.generate(
            batch["prompts"], # sharded inference engine share identical inputs
            sampling_params=self.train_sampling_params if train else self.test_sampling_params,
            use_tqdm=(self.rank == 0)
        )
        batch.update({
            "responses": [response.outputs[0].text for response in responses],
            "response_token_ids": [list(response.outputs[0].token_ids) for response in responses]
        }) # TODO: log response length
        
        batch = dispatch(batch, self.rollout_local_rank, self.args.tp_size)
        # only support outcome reward, RM should be served remotely if there is
        rewards = [
            self.reward_fn(response, answer)
            for response, answer in zip(batch["responses"], batch["answers"])
        ]
        batch["rewards"] = [
            (len(response_token_id) - 1) * [0] + [reward]
            for response_token_id, reward
            in zip(batch["response_token_ids"], rewards)
        ]

        batch = all_gather(batch)
        rewards = torch.Tensor([reward[-1] for reward in batch["rewards"]])
        if self.rank == 0 and not self.args.disable_wandb:
            wandb.log({
                f"reward/{'train' if train else 'test'}": rewards.mean().item()
            }, step=self.step)
        
        if not train:
            return
        
        rewards = rewards.view(-1, self.args.rollout_per_prompt)
        # compute advantage for REINFORCE. If use PPO, it will be replaced in `compute_gae`
        baselines = rewards.mean(-1, keepdim=True)
        advantages = (rewards - baselines).flatten().tolist()
        batch["advantages"] = [
            len(response_token_id) * [advantage]
            for response_token_id, advantage
            in zip(batch["response_token_ids"], advantages)
        ]

        return batch

    @torch.no_grad()
    def compute_log_probs(self, batches, prefix):
        model = self.model if prefix == "old" else self.ref_model
        load_fsdp_model_to_gpu(model)

        for batch in (tqdm(batches, desc=f"Step {self.step}, compute {prefix} log probs") if self.rank == 0 else batches):

            logits = model( # TODO: add ring attn in forward func
                batch["states"],
                position_ids=batch["position_ids"],
                use_cache=False
            ).logits

            batch[f"{prefix}_logps"] = torch.gather(
                logits.log_softmax(-1),
                dim=-1,
                index=batch["actions"].unsqueeze(-1)
            ).squeeze(-1) * batch["action_mask"]

            # add KL terms
            if prefix == "ref" and self.args.kl_type == "token":
                batch["rewards"] -= self.args.kl_coef * compute_kl_term(
                    batch["old_logps"], batch["ref_logps"], self.args.kl_estimator
                )
        
        offload_fsdp_model_to_cpu(model)

    @torch.no_grad()
    def compute_gae(self, batches):
        load_fsdp_model_to_gpu(self.critic)
        load_fsdp_optimizer(self.critic_optimizer, self.device)

        self.critic.eval()
        for batch in (tqdm(batches, desc=f"Step {self.step}, compute GAE") if self.rank == 0 else batches):
            values = self.critic( # TODO: add ring attn in forward func
                batch["states"],
                position_ids=batch["position_ids"],
                use_cache=False
            ).logits.squeeze(-1)

            # delta_t = r_t + \gamma * V(s_{t+1}) - V(s_t)
            # if s_{t+1} is a terminal state, V(s_{t+1}) = 0
            next_values = torch.cat(
                (values[:, 1:], torch.FloatTensor([[0]]).to(self.device)),
            dim=-1)
            delta = batch["rewards"] + self.args.gamma * batch["eos_mask"] * next_values - values

            # A_t = \delta_t + \gamma * \lambda * A_{t+1}
            # if s_{t+1} is a terminal state, A_{t+1} = 0
            # TODO: Prepare for ring attn: Shuffle into zigzag will shuffle the order of delta_t
            gae, reversed_gaes = 0, []
            for t in reversed(range(values.shape[-1])):
                gae = delta[0, t] + self.args.gamma * self.args.lamda * batch["eos_mask"][0, t] * gae
                reversed_gaes.append(gae)
            gaes = reversed_gaes[::-1]

            batch["advantages"] = torch.FloatTensor([gaes]).to(self.device)
            batch["returns"] = batch["advantages"] + values

    def update_critic(self, batches):

        self.critic.train()
        total_response_tokens = sum([batch["action_mask"].sum() for batch in batches])
        for batch in (tqdm(batches, desc=f"Step {self.step}, update critic") if self.rank == 0 else batches):

            values = self.critic(
                batch["states"],
                position_ids=batch["position_ids"],
                use_cache=False
            ).logits.squeeze(-1)
            loss = ((values - batch["returns"]).pow(2) * batch["action_mask"]).sum() / total_response_tokens # TODO: log loss
            loss.backward()
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        offload_fsdp_model_to_cpu(self.critic)
        offload_fsdp_optimizer(self.critic_optimizer)

    def update_actor(self, batches: List[Dict[str, torch.LongTensor]]):
        load_fsdp_model_to_gpu(self.model)
        load_fsdp_optimizer(self.optimizer, self.device)

        total_response_tokens = sum([batch["action_mask"].sum() for batch in batches])
        total_responses = sum([(batch["eos_mask"] == 0).sum() for batch in batches])
        for batch in (tqdm(batches, desc=f"Step {self.step}, update actor") if self.rank == 0 else batches):

            logits = self.model( # TODO: add ring attn in forward func
                batch["states"],
                position_ids=batch["position_ids"],
                use_cache=False
            ).logits

            logps = torch.gather(
                logits.log_softmax(-1),
                dim=-1,
                index=batch["actions"].unsqueeze(-1)
            ).squeeze(-1)
            logp_ratio = (logps - batch["old_logps"]).exp()
            loss = - (batch["advantages"] * logp_ratio * batch["action_mask"]).sum() / total_response_tokens

            if self.args.kl_type == "loss":
                seq_logps = compute_seq_logps(
                    logps, batch["eos_mask"]
                )
                ref_seq_logps = compute_seq_logps(
                    batch["ref_logps"], batch["eos_mask"]
                )
                kl_term = compute_kl_term(
                    seq_logps, ref_seq_logps, self.args.kl_estimator
                )
                loss += self.args.kl_coef * kl_term() / total_responses

            loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def save(self):

        load_fsdp_model_to_gpu(self.model) # or the params cannot be summoned
        with FSDP.summon_full_params(
            self.model,
            offload_to_cpu=True,
            rank0_only=True,
            writeback=False
        ):
            if self.rank == 0:
                self.tokenizer.save_pretrained(f"{self.args.save_path}/{self.args.experiment_name}/step{self.args.step}")
                self.model.module.save_pretrained(f"{self.args.save_path}/{self.args.experiment_name}/step{self.args.step}")
        offload_fsdp_model_to_cpu(self.model)

    def train(self):

        self.model.train()

        self.step = 0
        # for batch in self.test_dataloader:
        #     self.rollout(batch, False)
    
        for epoch in range(self.args.n_epochs):
            self.sampler.set_epoch(epoch)
            for batch in self.train_dataloader:

                batch = self.rollout(batch, True)
                batches = self.prepare_update(batch)

                self.compute_log_probs(batches, "old")
                if hasattr(self, "ref_model"):
                    self.compute_log_probs(batches, "ref")

                if hasattr(self, "critic"):
                    self.compute_gae(batches)
                    self.update_critic(batches)

                self.update_actor(batches)
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
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--ring_attn_size", type=int, default=1)
    parser.add_argument("--ring_head_stride", type=int, default=1)

    parser.add_argument(
        "--critic_model_name", type=str, default=None,
        help="PPO will be used if not None, REINFORCE otherwise."
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--lamda", type=float, default=1.0,
        help="Hyper-parameter of generalized advantage estimator. See https://arxiv.org/pdf/1506.02438."
    )

    parser.add_argument(
        "--kl_coef", type=float, default=0.0,
        help="Reference model will be used if larger than zero."
    )
    parser.add_argument(
        "--kl_type", choices=["reward", "loss"],
        help="If `reward`, KL term will be added into the reward of each token. If `loss`, KL term will be added into the PPO surrogant objective."
    )
    parser.add_argument(
        "--kl_estimator", choices=["k1", "k2", "k3"],
        help="The estimator of KL divergence. See http://joschu.net/blog/kl-approx.html."
    )

    parser.add_argument("--disable_gradient_checkpointing", action="store_true")

    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--max_prompt_length", type=int)
    parser.add_argument("--max_response_length", type=int)
    parser.add_argument("--test_temperature", type=float, default=0.0)

    parser.add_argument("--reward_fn_path", type=str)

    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument(
        "--batch_size", type=int,
        help="The number of prompts each rollout"
    )
    parser.add_argument(
        "--rollout_per_prompt", type=int,
        help="The number of responses for each prompt"
    )
    parser.add_argument("--max_length_per_device", type=int)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--test_freq", type=int, default=-1)
    parser.add_argument("--save_freq", type=int, default=-1)
    parser.add_argument("--save_path", type=str, default="ckpts")
    args = parser.parse_args()

    _, _, world_size = initialize_global_process_group()
    device_mesh = dist.device_mesh.init_device_mesh("cuda", mesh_shape=(world_size,))

    trainer = Trainer(args, device_mesh)
    trainer.train()

if __name__ == "__main__":
    main()

# TODO: support multiple update per rollout
# TODO: support RingAttention