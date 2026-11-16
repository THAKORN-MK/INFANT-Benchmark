# INFANT Benchmark — Sinc-Mamba-KAN

โครงการนี้ศึกษาวิธี Sinc-Mamba-KAN จากบทความของ Mohammed Tawfik, IEEE Access, DOI [10.1109/ACCESS.2026.3733522](https://doi.org/10.1109/ACCESS.2026.3733522) ด้วย implementation ที่พัฒนาตามบทความ และเตรียมการเปรียบเทียบกับ `D:\INFANT CRY`

## สถานะปัจจุบัน: เทรนครั้งที่ 1 ครบ 5 folds แล้ว

ผู้ใช้เทรน run `20261116T193124Z_05d86d55d2` ครบ 5 folds โดยใช้ InfantCry-1214 แล้ว คะแนนเฉลี่ย Accuracy 90.12% และ Macro F1 86.34% ดู [รายงานผลครั้งที่ 1](reports/run_20261116T193124Z_05d86d55d2/REPORT.md) ซึ่งมีผลรายคลาส ภาพ และข้อจำกัด

ผลตรวจเบื้องต้นพบไฟล์ซ้ำข้าม tier และ label ขัดแย้ง 3 กลุ่ม โค้ดเลือกประเมินทีละ tier โดยบันทึกข้อมูลดังกล่าวและใช้ parent folder/metadata เป็น label ตาม README ของ dataset พร้อมจัดไฟล์ byte-identical ใน tier เดียวกันให้อยู่ fold เดียวกัน ไม่ใช่การพิสูจน์ว่าเป็น infant-disjoint หรือทำซ้ำ protocol เดิมของบทความ

ตรวจ dataset ใหม่แบบอ่านอย่างเดียวใน WSL:

```bash
cd "/mnt/d/INFANT Benchmark"
source /home/adminuser/.venvs/audio-ml-gpu/bin/activate
export PYTHONPATH="$PWD/src"
python -m infant_benchmark.manifest --root "data/InfantCry-DBL A Two-Tier Annotated Corpus of Infan/Datasets"
```

ค่า exit code 1 ที่พบขณะนี้หมายถึงมีรายการที่ต้องตรวจ เช่น label ขัดแย้ง ไม่ได้หมายความว่าไฟล์เสียงต้นฉบับถูกแก้ไข คำสั่งนี้เป็นการตรวจสอบ ไม่ใช่การเทรน

การฝึกจริงเสร็จแล้วหนึ่ง run; feature extractor เป็น 213 ค่าแบบ proxy ตามข้อสมมติ ไม่ใช่ schema ที่ยืนยันจากผู้เขียน และผลยังไม่ใช่ external validation

ข้อจำกัดก่อนฝึก: ยังไม่พบ source code ทางการ รายละเอียดสำคัญในบทความยังไม่ครบสำหรับยืนยัน implementation ที่เหมือนต้นฉบับ และแม้ข้อมูลชุดที่เพิ่งใส่มี metadata แล้ว ความหมายของ infant ID ยังไม่ยืนยัน รวมถึงมีไฟล์ซ้ำและชื่อ label ที่ต้องพิจารณาตามนโยบายที่บันทึกไว้ ดู [รายงานการทำซ้ำเดิม](docs/replication_audit.md) และ [ข้อมูลอัปเดต](docs/paper_methods_th.md)

ผู้ใช้เป็นผู้สั่งเทรนเองใน environment แยก ไม่ติดตั้งทับ `audio-ml-gpu` เดิม ผลเทียบกับ `D:\INFANT CRY` ในรายงานเป็นเพียงการเทียบเบื้องต้น เพราะใช้ข้อมูลและ folds ต่างกัน

## ไฟล์ที่มี

| รายการ | หน้าที่ |
|---|---|
| `docs/replication_audit.md` | หลักฐานจากบทความ สิ่งที่ขาด การตรวจข้อมูล และคำถามที่ต้องตัดสินใจก่อนทำต่อ |
| `configs/paper_full_capacity.json` | สมุดบันทึกค่าที่บทความระบุและช่องที่ยังไม่ทราบ **ไม่ใช่ config สำหรับ train** |
| `src/infant_benchmark/audit.py` | ตรวจโครงสร้างและ SHA-256 ของไฟล์ WAV แบบอ่านอย่างเดียว |
| `src/infant_benchmark/manifest.py` | ตรวจ metadata กับไฟล์จริง รองรับชื่อโฟลเดอร์ที่สะกดต่างกันและตรวจ hash ข้าม tier |
| `src/infant_benchmark/audio.py`, `features.py`, `fold_features.py` | เตรียมเสียงและสกัด feature proxy; fit scaler/selector เฉพาะข้อมูลฝึกที่ส่งให้ |
| `src/infant_benchmark/train.py`, `training.py`, `model.py` | คำสั่งฝึก, 5-fold และโมเดลแบบ paper-inspired; ผู้ใช้เป็นผู้สั่งรันเอง |
| `src/infant_benchmark/reporting.py`, `scripts/build_run_report.py` | ตรวจความสอดคล้องของ run และสร้างภาพรายงานในโฟลเดอร์ใหม่ |
| `docs/paper_methods_th.md` | อธิบายวิธีหลัก baseline การประเมิน และความแตกต่างจากสิ่งที่โครงการทำได้แล้ว |
| `reports/run_20261116T193124Z_05d86d55d2/REPORT.md` | รายงานครั้งที่ 1 พร้อมผลราย fold รายคลาส ภาพ และแผนขั้นต่อไป |
| `docs/PAIRED_COMPARISON_PLAN.md` | แผนเทียบกับ Stage 2 main ด้วย records และ fold ชุดเดียวกัน โดยยังไม่เทรน |
| `tests/test_dataset_audit.py` | ทดสอบเครื่องมือตรวจด้วยไฟล์จำลอง ไม่เทรนโมเดล |
| `data/README.md` | แนวทางเตรียม dataset และ metadata โดยไม่แตะต้นฉบับ |

## คำสั่งตรวจข้อมูลเดิมแบบอ่านอย่างเดียวใน WSL

เปิด Ubuntu ผ่าน VS Code แล้วคัดลอกคำสั่งต่อไปนี้ สามารถใช้ Python 3.10 ใน environment เดิมได้ เครื่องมือนี้ใช้เฉพาะ Python standard library และไม่ import TensorFlow หรือ PyTorch:

```bash
cd "/mnt/d/INFANT Benchmark"
source /home/adminuser/.venvs/audio-ml-gpu/bin/activate
export PYTHONPATH="$PWD/src"

python -m infant_benchmark.audit --root "/mnt/d/INFANT CRY/data_set_dbl"
```

ผล JSON แสดงบน Terminal เท่านั้น: จำนวน WAV ตามโฟลเดอร์, รูปแบบ WAV, ชื่อไฟล์ที่มี suffix ไม่ตรงกับการแมปชื่อโฟลเดอร์, กลุ่มไฟล์ที่มี SHA-256 ตรงกันทั้งไฟล์ และ metadata ที่พบใน root ที่ระบุ เครื่องมือ **ไม่เขียนหรือย้ายไฟล์ใน dataset**

- SHA-256 ในที่นี้เป็น hash ของไฟล์ WAV ทั้งไฟล์: ไฟล์เสียงที่ฟังเหมือนกันแต่ header หรือ encoding ต่างกันอาจตรวจไม่พบ
- สถานะ tier และ infant identity เป็น `not_verified` เสมอ เครื่องมือไม่ได้อ้างว่ายืนยัน metadata หรือ label ของบทความได้
- Exit code `0` หมายถึงอ่านไฟล์ที่ตรวจได้โดยไม่มีข้อผิดพลาดด้านการอ่าน ไม่ได้หมายความว่า dataset ผ่านเกณฑ์การวิจัย; `2` หมายถึงไม่พบ directory หรือเกิดข้อผิดพลาดเริ่มต้น; `3` หมายถึงพบข้อผิดพลาดในการอ่าน hash หรือ WAV header
- โฟลเดอร์ `not_baby` ถูกนับเป็น `other_wav_count` แต่ไม่รวมในห้าคลาสของบทความ

ทดสอบเครื่องมือด้วยข้อมูลจำลอง:

```bash
cd "/mnt/d/INFANT Benchmark"
python -m unittest discover -s tests -v
```

## ก่อนรันการฝึก

อ่าน [รายงานผลและข้อจำกัด](reports/run_20261116T193124Z_05d86d55d2/REPORT.md) ก่อนวางแผนรอบใหม่ คำสั่งฝึกต้องเพิ่ม `--execute`, `--acknowledge-approximation` และ `--acknowledge-data-issues` ผู้ใช้เป็นผู้เริ่มเองเท่านั้น คำสั่งที่ไม่ใส่ `--execute` จะเป็น dry run ไม่โหลดข้อมูลหรือสร้างไฟล์
