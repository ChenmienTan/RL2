import torch
import torch.distributed as dist

def count_total_actions(minibatches, device_mesh):
        
    total_actions = sum(
        [minibatch["action_mask"].sum() for minibatch in minibatches]
    )
    total_actions = torch.Tensor(
        [total_actions]
    ).to(torch.cuda.current_device())
    for mesh_name in ["sp", "dp"]:
        dist.all_reduce(
            total_actions,
            op=dist.ReduceOp.SUM,
            group=device_mesh[mesh_name].get_group()
        )
    return total_actions.to("cpu").item()