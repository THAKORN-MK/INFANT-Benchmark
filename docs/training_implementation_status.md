# สถานะโค้ดเทรน Sinc-Mamba-KAN แบบอ้างอิงบทความ

## สิ่งที่เพิ่ม

- `model.py` มี SincConv, convolution, Mamba wrapper และ Gaussian RBF-KAN ซึ่งเป็น implementation ใหม่ ไม่ใช่ source ของผู้เขียน Gaussian RBF, การกำหนดขอบเขต sinc/padding/ค่าเริ่มต้น และรายละเอียด feature 213 ค่าเป็นข้อสมมติ
- `splits.py` สร้าง 5 folds โดยจัดไฟล์ที่มี SHA-256 ตรงกันให้อยู่กลุ่มเดียวกัน ใช้ข้อมูลทีละ tier และปฏิเสธ group ที่มี label ขัดแย้งภายใน tier ซึ่งต่างจาก clip-level 5-fold ในบทความและไม่พิสูจน์ infant-disjoint
- `training.py` มีวงรอบฝึก กำหนด 80 epochs ล่วงหน้าเป็นค่าเริ่มต้น และ fit scaler/MI selector เฉพาะ train ของ fold ไม่ใช้ held-out fold เพื่อเลือก epoch มีการบันทึก split, configuration, dependency/source hashes, ประวัติ loss, model, ตัวแปลง feature, prediction และ metrics แยกตาม run/fold
- `train.py` ค่าเริ่มต้นเป็น dry run ต้องให้ผู้ใช้เพิ่ม `--execute`, `--acknowledge-approximation` และ `--acknowledge-data-issues` จึงจะเริ่มทำงานจริง มีการตรวจ CUDA/Mamba ด้วยสัญญาณจำลองก่อนอ่านข้อมูลฝึกและสร้าง run

## ขอบเขตและสิ่งที่ต้องยืนยัน

ผู้ใช้ได้รันฝึกจริงครบ 5 folds แล้วใน run `20261116T193124Z_05d86d55d2`; ผู้ช่วยไม่ได้เป็นผู้สั่งฝึกหรือทำ optimizer step เอง ผลและการตรวจสอบอยู่ใน [รายงานครั้งที่ 1](../reports/run_20261116T193124Z_05d86d55d2/REPORT.md) การทดสอบส่วนประกอบกับข้อมูลจำลองไม่ได้ยืนยันความเท่ากันกับบทความ

ข้อมูลมี 180 hash groups ซ้ำข้าม tier รวม 3 groups ที่ label ขัดกัน โค้ดนี้ใช้ tier เดียวต่อ run ไม่ใช้เพื่อรายงาน cross-dataset; ภายใน tier เดียวกันผลนับทุกคลิปแม้เป็นไฟล์ซ้ำ ไม่ใช่จำนวนตัวอย่างอิสระ อัตลักษณ์ทารกไม่ได้รับการพิสูจน์

ตัวเลข noise 0.005, feature noise 0.01, shift ไม่เกิน 1,600 samples และ PolyLoss epsilon 1.0 เป็นสมมติฐาน ไม่ใช่ค่าที่ยืนยันได้จากบทความ และจะถูกบันทึกใน config ของ run การเปรียบเทียบกับ `D:\INFANT CRY` ต้องตรวจข้อมูลและ split ให้ตรงกัน ไม่ใช่เปรียบเทียบคะแนนข้าม protocol โดยตรง

## Environment

สร้าง `/home/adminuser/.venvs/infant-benchmark` แยกจาก `audio-ml-gpu` และตรวจบนเครื่องนี้แล้ว: PyTorch `2.9.1+cu128`, Mamba `2.3.2.post1`, NVIDIA GeForce RTX 5070 Ti; synthetic forward และ backward ผ่านโดยไม่มี optimizer step และ `pip check` ผ่าน `nvidia-cutlass-dsl` ตรึงที่ 4.5.1 เนื่องจาก 4.8.1 ไม่เข้ากับ Quack ที่ติดตั้ง ผู้ใช้รันฝึกจริงครบตามผลที่บันทึกแล้ว

## คำสั่งให้ผู้ใช้เริ่มเอง

อ่านข้อจำกัดข้างต้นก่อน คำสั่งนี้ใช้ Dataset 2 ทั้ง 1,214 คลิป เป็น 5-fold จึงไม่ใช่ชุดทดสอบภายนอกหรือ held-out set ที่อิสระจากงานเดิม หากคลิปซ้อนกับชุดทดสอบเดิม ผลนี้ไม่ใช่การยืนยันด้วยข้อมูลใหม่ คำสั่งใช้ 5 โมเดล สูงสุด 80 epochs ต่อ fold และจะไม่เขียนทับ run เก่า

```bash
cd "/mnt/d/INFANT Benchmark"
source /home/adminuser/.venvs/infant-benchmark/bin/activate
export PYTHONPATH="$PWD/src"

time python -m infant_benchmark.train \
  --dataset-root "data/InfantCry-DBL A Two-Tier Annotated Corpus of Infan/Datasets" \
  --tier InfantCry-1214 \
  --epochs 80 \
  --batch-size 16 \
  --seed 42 \
  --execute \
  --acknowledge-approximation \
  --acknowledge-data-issues
```

ช่วงต้นจะตรวจ GPU, อ่าน metadata และสกัด features อาจยังไม่เห็น epoch จนกว่าการสกัดข้อมูลเสร็จ หากเกิด error ให้เก็บข้อความทั้งหมดไว้ตรวจสอบ; run ไม่สำเร็จจะยังไม่มี `summary.json` และไม่ควรนับเป็นผลสำเร็จ ผลลัพธ์อยู่ใน `runs/<run_id>/` แต่ละ fold มี model, split, transform, loss, predictions และ metrics ส่วน `summary.json` จะปรากฏเมื่อครบ 5 folds เท่านั้น
