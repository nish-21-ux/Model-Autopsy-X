import os
import tarfile
import zstandard

archive = "trained_model/best-checkpoint.tgz"
output_dir = "trained_model/extracted"

os.makedirs(output_dir, exist_ok=True)

with open(archive, "rb") as f:
    reader = zstandard.ZstdDecompressor().stream_reader(f)

    with tarfile.open(fileobj=reader, mode="r|") as tar:
        for member in tar:
            if member.isfile():
                tar.extract(member, output_dir)

print("MODEL CHECKPOINT EXTRACTED SUCCESSFULLY!")