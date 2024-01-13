"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU small debug run, example:
$ python -m train.py --compile=False --eval_iters=10 --batch_size=8

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial
import numpy as np
import torch
from model import Transformer, ModelArgs, RewardModel
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from preference import Task
from export import model_export
from tokenizer import Tokenizer
from preference import get_tokenizer_model_path

# -----------------------------------------------------------------------------
# I/O
eval_interval = 1
log_interval = 1
eval_iters = 10
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "resume"  # 'scratch' or 'resume'
pretrained_checkpoint = 'out15M/stories15M.pt'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = "llamac"
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# data
batch_size = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 256
vocab_source = "llama2" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size = 32000 # the Llama 2 tokenizer has 32K tokens
# model
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
multiple_of = 32
dropout = 0.0
# adamw optimizer
gradient_accumulation_steps = 4  # used to simulate larger batch sizes
learning_rate = 5e-4  # max learning rate
max_iters = 100000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = False  # whether to decay the learning rate
warmup_iters = 1000  # how many steps to warm up for
# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "float16"  # float32|bfloat16|float16
compile = False  # use PyTorch 2.0 to compile the model to be faster
# loss hparams
beta = 0.1
IPO_tau_parameter=0.1
loss_type = "DPO"
set_type="set_1"
# entropy
max_new_tokens = 256
temperature = 1.0
top_k = 300
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------
out_dir = f"out_{loss_type}_beta_{str(beta).replace('.','_')}_{set_type}_IPO_{str(IPO_tau_parameter).replace('.','_')}_lr_{str(learning_rate).replace('.','_')}"
print("Out dir: ", out_dir)
# fixing some hyperparams to sensible defaults
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# validating checks
assert vocab_source in ["llama2", "custom"]
assert vocab_source == "custom" or vocab_size == 32000, "The vocab from Meta has 32K tokens"

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
summary_writer = SummaryWriter(log_dir=os.path.join(out_dir, "logs"))
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# task-specific setup
iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    vocab_size=vocab_size,
    vocab_source=vocab_source,
    device=device,
    num_workers=0,
    label_name=set_type
)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = ModelArgs(**model_args)
    model_ref = Transformer(gptconf)
    model_ref.eval()
    model = Transformer(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {pretrained_checkpoint}")
    # resume training from a checkpoint.
    checkpoint = torch.load(pretrained_checkpoint, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    # iter_num = checkpoint["iter_num"]
    # best_val_loss = checkpoint["best_val_loss"]
    model_ref = Transformer(gptconf)
    model_ref.load_state_dict(state_dict)
    model_ref.eval()
model_ref.to(device)
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
# if init_from == "resume" and "optimizer" in checkpoint:
#     optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0
    model_ref = torch.compile(model_ref)

# wrap model into DDP container
if ddp:
    # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
    # construction time since NCCL does not support `ComplexFloat`
    prefix = "_orig_mod." if compile else ""
    model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
    model = DDP(model, device_ids=[ddp_local_rank])
    model_ref._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
    model_ref = DDP(model_ref, device_ids=[ddp_local_rank])

# Starting prompt
tokenizer_model = get_tokenizer_model_path(vocab_size=0)
enc = Tokenizer(tokenizer_model=tokenizer_model)

# encode the beginning of the prompt
start_ids = enc.encode("", bos=True, eos=False)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
x = x.repeat((batch_size, 1))

# helps estimate an arbitrarily accurate loss over either split using many batches
def get_log_prob_of_stories(tokens, targets, model, story_length):
    logits =  F.softmax(model(tokens), dim=-1)
    log_prob_tokens = torch.log(torch.gather(logits, -1, torch.unsqueeze(targets, 1)))
    log_prob_tokens = log_prob_tokens * story_length[:,1:]
    log_prob_of_stories = torch.sum(log_prob_tokens, dim=-1)
    return log_prob_of_stories

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            story1, story2, story1_l, story2_l  = next(batch_iter)
            with ctx:
                pie_theta_w = get_log_prob_of_stories(story1[:,:-1], story1[:,1:], model, story1_l)
                pie_theta_l = get_log_prob_of_stories(story2[:,:-1], story2[:,1:], model, story2_l)
                pie_ref_w = get_log_prob_of_stories(story1[:,:-1], story1[:,1:], model_ref, story1_l)
                pie_ref_l = get_log_prob_of_stories(story2[:,:-1], story2[:,1:], model_ref, story2_l)
                pi_logratios = pie_theta_w - pie_theta_l
                ref_logratio = pie_ref_w - pie_ref_l
                if loss_type == "DPO":
                    loss = get_dpo_loss(pi_logratios, ref_logratio)
                if loss_type == "IPO":
                    loss = get_ipo_loss(pi_logratios, ref_logratio)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def estimate_entropy():
    model.eval()
    entropy = []
    for _ in range(eval_iters):
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        story, logits = model.sample(x, max_new_tokens, temperature, top_k)
        log_prob_tokens = F.cross_entropy(logits, story.squeeze(), reduction="none")
        entropy.append(log_prob_tokens.sum().item())
    model.train()
    return -np.mean(entropy)
        

def get_dpo_loss(pi_logratios, ref_logratio):
    return -F.logsigmoid(beta * (pi_logratios - ref_logratio)).sum()

def get_ipo_loss(pi_logratios, ref_logratio):
    return torch.square(pi_logratios - ref_logratio - 1/(2*IPO_tau_parameter)).sum()

# def get_loss(pie_theta_w, pie_theta_l, pie_ref_w, pie_ref_l):
#     pie_log_ratios = torch.log(pie_theta_w) - torch.log(pie_theta_l)
#     ref_log_ratios = torch.log(pie_ref_w) - torch.log(pie_ref_l)
#     losses = -F.logsigmoid(beta * (pie_log_ratios - ref_log_ratios))
#     print(losses)
#     return losses.mean()


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
train_batch_iter = iter_batches(split="train")
story1, story2, story1_l, story2_l  = next(train_batch_iter)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        entropy = estimate_entropy()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, entropy: {entropy:.4f}")
        if wandb_log:
            try:
                wandb.log(
                    {
                        "iter": iter_num,
                        "tokens": iter_num * tokens_per_iter,
                        "loss/train": losses["train"],
                        "loss/val": losses["val"],
                        "lr": lr,
                        "mfu": running_mfu * 100,  # convert to percentage
                    }, step = iter_num
                )
            except Exception as e:
                print(f"logging to wandb failed: {e}")
        summary_writer.add_scalar("Loss/train",losses["train"], iter_num)
        summary_writer.add_scalar("Loss/eval",losses["val"], iter_num)
        summary_writer.add_scalar("Entropy", entropy, iter_num)
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                model_export(raw_model, os.path.join(out_dir, "model.bin"), version=0)
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx:
            pie_theta_w = get_log_prob_of_stories(story1[:,:-1], story1[:,1:], model, story1_l)
            pie_theta_l = get_log_prob_of_stories(story2[:,:-1], story2[:,1:], model, story2_l)
            pie_ref_w = get_log_prob_of_stories(story1[:,:-1], story1[:,1:], model_ref, story1_l)
            pie_ref_l = get_log_prob_of_stories(story2[:,:-1], story2[:,1:], model_ref, story2_l)
            pi_logratios = pie_theta_w - pie_theta_l
            ref_logratio = pie_ref_w - pie_ref_l
            if loss_type == "DPO":
                loss = get_dpo_loss(pi_logratios, ref_logratio)
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        story1, story2, story1_l, story2_l  = next(train_batch_iter)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
summary_writer.flush()
summary_writer.close()
if ddp:
    destroy_process_group()
