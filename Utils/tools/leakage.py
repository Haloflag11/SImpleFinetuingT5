from tqdm import tqdm


def check_data_leakage(train_loader, val_loader):
    """基于input_ids和labels检查数据泄露"""

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    train_samples = set()
    val_samples = set()

    # 检查训练集
    for i in tqdm(range(len(train_dataset))):
        item = train_dataset[i]

        # 关键修正：直接使用张量的哈希值或字符串表示
        # 方法1：使用元组哈希（推荐）
        input_ids_tuple = tuple(item['input_ids'].tolist())
        labels_tuple = tuple([x for x in item['labels'].tolist() if x != -100])  # 过滤padding

        sample_key = (input_ids_tuple, labels_tuple)
        train_samples.add(sample_key)

    # 检查验证集
    for i in range(len(val_dataset)):
        item = val_dataset[i]

        input_ids_tuple = tuple(item['input_ids'].tolist())
        labels_tuple = tuple([x for x in item['labels'].tolist() if x != -100])

        sample_key = (input_ids_tuple, labels_tuple)
        val_samples.add(sample_key)

    # 检查重叠
    overlap = train_samples.intersection(val_samples)

    print(f"训练集唯一样本数: {len(train_samples)}")
    print(f"验证集唯一样本数: {len(val_samples)}")
    print(f"训练集与验证集重叠样本数: {len(overlap)}")

    if overlap:
        print("⚠️ 发现数据泄露！重叠样本的token IDs:")
        for sample in list(overlap)[:3]:
            print(f"Input IDs: {sample[0][:10]}...")  # 显示前10个token
            print(f"Labels: {sample[1][:10]}...")
            print("---")

    return len(overlap) == 0
