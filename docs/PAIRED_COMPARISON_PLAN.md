# แผนเปรียบเทียบแบบใช้ข้อมูลและ fold เดียวกันกับ INFANT CRY

## เป้าหมายและสถานะ

เปรียบเทียบโมเดลหลัก Stage 2 ของ `D:\INFANT CRY` กับ Sinc-Mamba-KAN แบบ paper-inspired ของ `D:\INFANT Benchmark` โดยใช้ OOF records และ fold assignment ชุดเดียวกันเพื่อให้เห็นความต่างของ **สอง pipeline** ภายใต้ข้อมูลประเมินที่ตรงกัน

สถานะ: **เป็นแผน ยังไม่ได้สร้าง manifest ใหม่หรือเทรนรอบเปรียบเทียบ** ผลที่ทำภายหลังต้องเรียก exploratory เนื่องจากทีมได้เห็นผล benchmark และผลของงานเดิมแล้ว แผนนี้ไม่ได้เป็นการลงทะเบียนล่วงหน้าก่อนเห็นผลทั้งหมด

## 1. ชุดข้อมูลที่เสนอ

ใช้ไฟล์ [fold_assignments.csv ของโมเดลหลัก](<../../INFANT CRY/Models_dbl/Main/runs/20260821T164332Z_490383ff/fold_assignments.csv>) และผล OOF เดิมเป็นจุดอ้างอิง:

| หลักฐาน | จำนวน | สิ่งที่ตรวจพบ |
|---|---:|---|
| Original OOF records ของโมเดลหลัก | 1,045 | มี record ID, SHA-256, label และ fold อยู่แล้ว |
| ตรงกับ Tier 2 ใน run benchmark เดิม | 917 | SHA-256 และ label ที่แมปตรงกัน |
| ไม่พบใน Tier 2 แต่ตรงโฟลเดอร์ `dunstan  337` | 128 | ไฟล์ขึ้นต้น `dunstan-`; SHA-256 และ label ที่แมปตรงกันทั้ง 128 |
| ตรงกับหนึ่งในสองกลุ่มข้างต้น | 1,045 | ต้องตรวจซ้ำและบันทึก manifest ก่อนเริ่มจริง |

จำนวนข้างต้นเป็นการเทียบจากหลักฐานปัจจุบัน ไม่ใช่การพิสูจน์ตัวตนทารก 1,045 ราย และไม่ได้รับประกันว่าเสียงที่ bytes ต่างกันไม่ซ้ำกัน ในการใช้งานจริงต้องตรวจแหล่งไฟล์และสิทธิ์ใช้ข้อมูลรายแหล่งอีกครั้ง

**สำคัญ:** Cohort นี้ผสม Tier 1 และ Tier 2 เพื่อจับคู่กับงานเดิม จึงตอบคำถามเรื่องการเปรียบเทียบกับโมเดลหลัก ไม่ใช่การทำซ้ำการประเมิน InfantCry-1214 ล้วนตามบทความ

## 2. ขั้นตอนเตรียม manifest — ยังไม่เทรน

1. อ่าน `fold_assignments.csv` และ `oof_predictions.csv` ของโมเดลหลักแบบ read-only เลือก original records 1,045 รายการตามหลักฐานที่ตรึงไว้
2. จับคู่แต่ละ record กับไฟล์เสียงด้วย SHA-256 และ mapping `belly_pain→eairh`, `burping→eh`, `discomfort→heh`, `hungry→neh`, `tired→owh`; ตรวจ label, จำนวน, path และความซ้ำ ถ้าไม่ตรงให้หยุด ไม่แก้ label อัตโนมัติ
3. คัดลอก **หมายเลข validation fold เดิมของแต่ละ record** ไปยัง manifest ใหม่ ตรวจ 5 folds ครบ, ทุก record ถูกประเมินครั้งเดียว และ hash กลุ่มเดียวกันไม่ข้ามฝั่งฝึก/ประเมิน
4. ตรวจว่าไม่มี hash ของชุด Final Test 303 records ของโมเดลหลักอยู่ใน cohort ที่นำมาฝึกหรือประเมิน paired OOF; เปรียบเทียบเฉพาะ hash/ID โดยไม่ใช้คะแนน Test เลือกโมเดล
5. บันทึก hash ของ manifest, input evidence, source/config, policy และรายงานเหตุการณ์ผิดปกติ ไม่ย้ายหรือแก้ไฟล์เสียง/ผล run ต้นฉบับ

## 3. การรันในอนาคต — ผู้ใช้เป็นผู้เริ่มเอง

ต้องปรับ benchmark ให้รองรับ manifest และ fold ที่ระบุ ไม่ให้สุ่ม fold ใหม่ และไม่ให้ใช้คลิปนอก manifest:

- ฝึก 5 โมเดลใหม่ตั้งแต่ต้น — หนึ่งโมเดลต่อ fold โดย held-out fold ตรงกับของโมเดลหลัก
- คำนวณ StandardScaler และ MI feature selection จาก training fold เท่านั้น; augmentation เฉพาะ training fold
- ตรึง recipe, seed และจำนวน epochs ก่อนรัน และรายงานว่าต่างจากโมเดลหลักอย่างไร; ห้ามใช้คะแนน held-out เลือก epoch ของ fold นั้น
- บันทึก predictions, labels, fold, hash, metrics และ config แยก run ไม่เขียนทับของเดิม
- **ผู้ช่วยห้ามเริ่มการเทรนเอง**; เมื่อโค้ดผ่านการตรวจและได้รับคำสั่งให้ดำเนินการ จะให้คำสั่งแก่ผู้ใช้เพื่อรันเอง

## 4. วิเคราะห์และรายงาน

1. ตรวจว่า prediction ของทั้งสองวิธีครบ 1,045 records เดียวกัน, hash/label/fold ตรงกันทุกตัว ก่อนรวมตาราง
2. คำนวณ Accuracy, Balanced Accuracy, Macro F1, Macro AUC และ per-class Precision/Recall/F1 ด้วยวิธีรวม OOF เดียวกัน พร้อม confusion matrix แยกสองโมเดล
3. รายงานทั้งผลราย fold, fold mean ± SD และ pooled OOF โดยระบุว่าสถิติเหล่านี้ต่างกันอย่างไร ความต่างของ model scores เป็น **ผลภายใน cohort เดียวกัน** ไม่ใช่หลักฐานการใช้งานจริงทั่วไป
4. หากวิเคราะห์ความไม่แน่นอนหรือทดสอบความต่าง ให้ระบุวิธีและหน่วยกลุ่มล่วงหน้าสำหรับการวิเคราะห์นั้น SHA-256 groups ไม่ใช่หลักฐานว่าเป็นทารกอิสระกัน และไม่ควรอ้างนัยสำคัญโดยไม่มีการทดสอบที่เหมาะสม

## 5. Final Test และข้อสรุป

ก่อนหน้านี้พบว่า 275 จาก 303 records ใน Final Test เดิมมีเนื้อหาตรงกับคลิปที่อยู่ใน benchmark 1,214-clip CV run จึงไม่ใช่ข้อมูลใหม่ที่ยังไม่เคยถูกใช้กับ benchmark นอกจากนี้ ผลเดิมถูกดูแล้ว ไม่ควรนำมาปรับค่าหรืออ้างเป็น external validation

หากต้องการข้อสรุปยืนยัน ควรกำหนด protocol ก่อนทดลองและจัดหาชุดประเมินอิสระที่ไม่เคยใช้ฝึก/เลือก โดยหากอ้างประสิทธิภาพกับทารกคนใหม่ ต้องมีข้อมูล infant/session identity ที่เชื่อถือได้และจัดกลุ่มอย่างเหมาะสม

## 6. หลักฐานและข้อแตกต่างที่ต้องแนบ

- [รายงาน benchmark 1,214 คลิป](../reports/run_20261116T193124Z_05d86d55d2/REPORT.md)
- [ผล OOF โมเดลหลัก](<../../INFANT CRY/Models_dbl/Main/runs/20260821T164332Z_490383ff/oof_metrics.json>)
- ความแตกต่าง: acoustic features, waveform length/rate, การ augment, class weighting, architecture, epoch policy, optimizer และ resource usage
- ไม่มี source ของผู้เขียน Sinc-Mamba-KAN ที่ยืนยันได้; โค้ด benchmark ใช้ 213-feature proxy และ Gaussian RBF-KAN ซึ่งยังไม่ยืนยันว่าเหมือนต้นฉบับ
