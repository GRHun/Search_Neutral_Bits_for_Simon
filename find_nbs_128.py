import simon128 as sm
import numpy as np
from os import urandom, cpu_count
import concurrent.futures

m = 2
WORD_SIZE = sm.WORD_SIZE()
MASK_VAL = 2 ** WORD_SIZE - 1

def make_structure(pt0, pt1, diff, neutral_bits):
    p0 = pt0.reshape(-1, 1)
    p1 = pt1.reshape(-1, 1)
    for i in neutral_bits:
        d = 1 << i
        d0 = d >> WORD_SIZE
        d1 = d & MASK_VAL
        p0 = np.concatenate([p0, p0 ^ d0], axis=1)
        p1 = np.concatenate([p1, p1 ^ d1], axis=1)
    
    p0b = p0 ^ diff[0]
    p1b = p1 ^ diff[1]
    return p0, p1, p0b, p1b

def gen_key(nr):
    key = np.frombuffer(urandom(8*m), dtype=np.uint64).reshape(m, -1)
    ks = sm.expand_key(key, nr)
    return ks

def gen_plain(n):
    pt0 = np.frombuffer(urandom(8*n), dtype=np.uint64)
    pt1 = np.frombuffer(urandom(8*n), dtype=np.uint64)
    return pt0, pt1

def process_batches(num_batches, batch_size, nr, in_diff, out_diff, tested_bit):
    """
    处理指定数量的batch，并返回累计的 total_pass 和 neutral_num
    """
    total_pass = 0
    neutral_num = 0
    for _ in range(num_batches):
        pt0, pt1 = gen_plain(batch_size)
        p0, p1, p0b, p1b = make_structure(pt0, pt1, diff=in_diff, neutral_bits=tested_bit)
        key = gen_key(nr)
        ct0a, ct1a = sm.encrypt((p0, p1), key)
        ct0b, ct1b = sm.encrypt((p0b, p1b), key)
        
        ct_diff0 = ct0a ^ ct0b  # shape: (batch_size, 2)
        ct_diff1 = ct1a ^ ct1b  # shape: (batch_size, 2)
        
        cond_full = (ct_diff0[:,0] == out_diff[0]) & (ct_diff1[:,0] == out_diff[1]) & \
                    (ct_diff0[:,1] == out_diff[0]) & (ct_diff1[:,1] == out_diff[1])
        cond_partial = (ct_diff0[:,0] == out_diff[0]) & (ct_diff1[:,0] == out_diff[1])
        
        total_pass += np.count_nonzero(cond_partial)
        neutral_num += np.count_nonzero(cond_full)
    return total_pass, neutral_num

def check_neutral_bit_batch(total_iters, batch_size, nr, in_diff, out_diff, tested_bit, num_workers=None):
    """
    使用多进程批量检测中性比特：
    - total_iters: 总迭代次数
    - batch_size: 每批生成的明文对数
    - tested_bit: 列表形式的待检测中性比特位置
    返回正确对(total_pass)和中性比特(neutral_num)
    """
    if num_workers is None:
        num_workers = cpu_count() - 8
    total_batches = total_iters // batch_size
    # 为了减少任务调度开销，将总批次数分块，每个任务处理 chunk_size 个batch
    chunk_size = total_batches // num_workers
    remainder = total_batches % num_workers

    tasks = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i in range(num_workers):
            # 将余数均摊给前面几个进程
            batches_for_worker = chunk_size + (1 if i < remainder else 0)
            tasks.append(executor.submit(process_batches, batches_for_worker, batch_size, nr, in_diff, out_diff, tested_bit))
        total_pass = 0
        neutral_num = 0
        for future in concurrent.futures.as_completed(tasks):
            tp, nn = future.result()
            total_pass += tp
            neutral_num += nn
    return total_pass, neutral_num

def check(f_path):
    nr = 6
    save_txt = f"{f_path}search_simon128_{nr}r.txt"
    with open(save_txt, "a") as f:
        in_diff = (0x0000000000080800, 0x0000000000220200)
        out_diff = (0x0, 0x200)
        print(f"\n\nFor simon128 - stage1: nr={nr}, in_diff={in_diff}, out_diff={out_diff}\nPr(path1) = 2**-20\n", file=f)
        
        neutrality = np.zeros(WORD_SIZE*2)
        TOTAL_ITERS = 2**30  # 总迭代次数
        BATCH_SIZE = 2**15   # 每批处理样本数
        
        for i in range(128):
            tested_bit = [i]
            total_pass, neutral_num = check_neutral_bit_batch(TOTAL_ITERS, BATCH_SIZE, nr, in_diff, out_diff, tested_bit)
            if total_pass == 0:
                print(f'i = {i}: passed_num == 0')
                print(f'i = {i}: passed_num == 0', file=f, flush=True)
            else:
                neutrality[i] = neutral_num / total_pass
                if neutrality[i] > 0.7:
                    print(f'i = {i}: neutral res is {neutrality[i]}')
                    print(f'i = {i}: neutral res is {neutrality[i]} ------------ here', file=f, flush=True)
                else:
                    print(f'i = {i}: neutral res is {neutrality[i]}')
                    print(f'i = {i}: neutral res is {neutrality[i]}', file=f, flush=True)
    print("检测结束。")

if __name__ == "__main__":
    fpath = r"/root/autodl-tmp/128/NBs128/"
    check(fpath)

