import json

def read_manifest_basic(filename):
    """Đọc manifest file cơ bản"""
    entries = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # Loại bỏ whitespace
            if line:  # Skip dòng trống
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error: {e}")

    return entries
def split_train_val_test(entries, train_ratio=0.97, val_ratio=0.01 ):
    """
    Chia dữ liệu thành tập train, val và test theo tỷ lệ nhất định.
    """
    total = len(entries)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_entries = entries[:train_end]
    val_entries = entries[train_end:val_end]
    test_entries = entries[val_end:]

    return train_entries, val_entries, test_entries
def save_manifest(entries, filename):
    """Lưu danh sách entries vào file manifest"""
    with open(filename, "w", encoding="utf-8") as f:
        for entry in entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")  # Thêm dòng mới sau mỗi entry
if __name__ == "__main__":
    # Ví dụ sử dụng
    manifest_file = "/kaggle/input/vivoice-16k-200h/audio_datasets/manifest.json"  # Thay đổi đường dẫn tới file manifest của bạn
    entries = read_manifest_basic(manifest_file)
    train_entries, val_entries, test_entries = split_train_val_test(entries)
    save_manifest(train_entries, "train_manifest.json")
    save_manifest(val_entries, "val_manifest.json")
    save_manifest(test_entries, "test_manifest.json")
    print(f"Train entries: {len(train_entries)}")
    print(f"Validation entries: {len(val_entries)}")
    print(f"Test entries: {len(test_entries)}")
    print("Manifest files created successfully.")