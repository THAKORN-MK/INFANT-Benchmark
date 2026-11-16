# รายงานตรวจสอบการทำซ้ำ Sinc-Mamba-KAN

วันที่ตรวจ: 16 พฤศจิกายน 2026  
สถานะ: **เตรียมการและตรวจหลักฐานเท่านั้น — ยังไม่พร้อม train**  
หลักฐานบทความ: PDF ที่ผู้ใช้แนบ, DOI [10.1109/ACCESS.2026.3733522](https://doi.org/10.1109/ACCESS.2026.3733522), SHA-256 ของไฟล์ PDF `5ca91eda760aec10c2a2e2f251a32853306559c27d7e3b8141a702119e9c89d9`

เลขหน้าในตารางเป็นเลขหน้าของไฟล์ PDF เริ่มนับจากหน้าแรก ไม่ใช่เลขหน้า 14209x ของวารสาร

## 1. วัตถุประสงค์และขอบเขต

ทำ benchmark ของวิธีหลักในบทความให้ใกล้ต้นฉบับที่สุด โดยไม่สับเปลี่ยนเป็น pipeline MFCC/Delta/Log-Mel ของโครงการ `D:\INFANT CRY` ผลของ benchmark ยังไม่ถูกสร้างและยังไม่ถูกนำไปเปรียบเทียบกับโครงการเดิม

ข้อมูลในบทความเป็นการจำแนก **ป้าย DBL ที่ผู้เชี่ยวชาญกำหนดจากเสียง** ไม่ใช่การยืนยันอาการหรือความต้องการทางสรีรวิทยาโดยตรง (PDF หน้า 3 หัวข้อ Scope of the Labels) จึงไม่ควรนำผลไปอ้างเป็นการวินิจฉัย

สถานะในตารางหมายถึงสถานะของข้อมูลในบทความและโครงการที่สร้างตอนนี้ ไม่ใช่ผลว่าทดลองทำซ้ำสำเร็จ:

| หัวข้อ | บทความระบุ | หลักฐาน | ช่องว่าง / สถานะการทำ |
|---|---|---|---|
| Dataset | Dataset 1 Dunstan-Core 337 clips; Dataset 2 InfantCry-1214 1,214 clips | PDF หน้า 3–5, ตาราง 1–2 | พบจำนวนไฟล์รวมที่มีรูปแบบคล้ายกันในโครงการเดิม แต่ยังไม่พิสูจน์ tier, version หรือไฟล์ตรงต้นฉบับ |
| Input waveform | 16 kHz, mono, 2 วินาที / 32,000 samples; truncate หรือ zero-pad และ peak-normalize | PDF หน้า 5 ข้อ III-B; Fig. 1 หน้า 3 | ยังไม่ระบุรายละเอียดทุกกรณี เช่น ตัด/เติมด้านใด, silent input และ exact resampling implementation ที่ใช้จริง; **ยังไม่ได้ implement** |
| Feature transform | FFT 512, hop 160 | PDF หน้า 5 ข้อ III-B | หน้าต่างและการตั้งค่าที่เหลือ รวมถึง dependency version ณ เวลาฝึก ยังไม่ยืนยัน; **ยังไม่ได้ implement** |
| Sinc front-end | 64 filters, kernel 251, stride 8, Hamming window; Fig. 1 แสดง output 64×4000 | PDF หน้า 3 รูป 1; หน้า 5 ข้อ III-C | padding, initialization, การบังคับช่วง cutoff และ library/implementation ไม่ครบสำหรับพิสูจน์ว่าตรงต้นฉบับ; **ยังไม่ได้ implement** |
| Conv refinement | ภาพระบุ BN → LeakyReLU(0.2) → MaxPool(4), ตามด้วย Conv 64→96 และ 96→128, kernel 5 และ MaxPool(4) | PDF หน้า 3 รูป 1; หน้า 5 ข้อ III-C | padding และรายละเอียด operator บางจุดยังต้องยืนยันจาก source; **ยังไม่ได้ implement** |
| Mamba | d_model 128, d_state 16, expansion 2; Fig. 1 ระบุ d_conv 4 และ residual | PDF หน้า 3 รูป 1; หน้า 5 และตาราง 3 หน้า 6 | ต้องทราบ package version, exact block และ parameterization เพื่อยืนยัน equivalence; **ยังไม่ได้ implement** |
| Acoustic features | 213 descriptors, fit scaler บน train แล้วเลือก mutual information top 100; projection 100→64 | PDF หน้า 5 ข้อ III-D; Algorithm 1 หน้า 7 | ยังไม่มีรายการชื่อ/ลำดับ/พารามิเตอร์ของ 213 ตัวครบ รวมถึงการคำนวณ F0/voicing และ spectral contrast; **ยังไม่ได้ implement** |
| KAN | รวม 128+64=192; KAN 192→64→5, grid 5, order 3, 8 bases, dropout 0.3 | PDF หน้า 3 รูป 1; หน้า 5 ข้อ III-E; ตาราง 3 หน้า 6 | รายละเอียด basis/grid/initialization และความสัมพันธ์ของคำอธิบาย Gaussian/RBF กับ B-spline ยังต้อง source ต้นฉบับ; **ยังไม่ได้ implement** |
| Training | weighted CE + PolyLoss-2, label smoothing 0.05, AdamW lr 0.001, weight decay 0.0001, cosine schedule, clip 2.0 | PDF หน้า 5 ข้อ III-F; Algorithm 1 หน้า 7; ตาราง 3 หน้า 6 | ไม่ระบุค่า epsilon ของ polynomial และรายละเอียด class-weight normalization/การรวม smoothing ที่เพียงพอ; **ยังไม่ได้ implement** |
| Full capacity | 80 epochs, Mamba d_state 16, expand 2; ตารางระบุ batch 16–32 | PDF หน้า 6 ตาราง 3; หน้า 7 ข้อ IV-C | batch ที่ใช้กับแต่ละ main run, seed ตัวเลข และ checkpoint/validation policy ยังไม่ยืนยัน; **ยังไม่ train** |
| Reduced compute | d_state 8, expand 1, 45 epochs สำหรับ ablation/cross-dataset บางส่วน | PDF หน้า 7 ข้อ IV-C | ไม่ใช้เป็น default แทน full capacity; **ยังไม่ implement** |
| Augmentation | Gaussian noise และ circular time shift ของ waveform; additive feature noise แบบ on-the-fly | PDF หน้า 5 ข้อ III-B, III-F; Algorithm 1 หน้า 7 | ค่า noise/shift, probability, class-specific sampling และลำดับที่ละเอียดกว่ายังไม่ครบ; **ยังไม่ได้ implement** |
| CV และ metrics | clip-level stratified 5-fold; infant-disjoint StratifiedGroupKFold สำหรับ Dataset 2; accuracy, balanced accuracy, macro-F1, OVR ROC AUC | PDF หน้า 5–6 ข้อ IV-A และหน้า 8 ข้อ V-B | ไม่มี split indices หรือ seed ที่ยืนยันแล้วในเครื่อง และ infant ID ยังไม่ผ่านการตรวจสอบ; **ยังไม่ได้แบ่ง fold** |
| Explainability | อ่าน sinc cutoffs; Integrated Gradients และ branch occlusion กับ model; SHAP/LIME กับ surrogate พร้อมวัด fidelity | PDF หน้า 9–12 ข้อ VI | ยังไม่มี trained model, surrogate หรือ implementation; **ยังไม่ได้ประเมิน** |
| เปรียบเทียบ/ablation | บทความมี baseline, ablation และ cross-dataset | PDF หน้า 7–9 และข้อ cross-dataset ในบทความ | เป็นงานเพิ่มเติมหลังยืนยัน protocol และ model; ยังไม่มีการเปรียบเทียบผลของเรา |

### ข้อสังเกตเกี่ยวกับ 213 features

บทความอธิบาย MFCC และ delta, chroma, spectral contrast, สถิติสเปกตรัม และ source/voicing descriptors แต่ยังไม่ได้ให้ schema รายชื่อ 213 ช่องและพารามิเตอร์ครบในส่วนที่ตรวจได้ การทำให้ vector มีความยาว 213 เพียงอย่างเดียว **ไม่พิสูจน์** ว่า feature ตรงกัน จึงไม่สร้าง feature เพิ่มแบบเดาให้ตัวเลขลงตัว

### จุดที่ต้องระวังเรื่องการประเมิน

บทความระบุ held-out fold สำหรับทดสอบ แต่รายละเอียดการเลือก epoch/checkpoint และวิธีใช้ validation สำหรับ main run ไม่ชัดเจนเพียงพอจากหลักฐานที่ตรวจ ห้ามนำ held-out fold ไปเลือกโมเดลโดยไม่เปิดเผย และห้ามกำหนดวิธีทดแทนแล้วอ้างว่าเป็น protocol เดิม ต้องขอข้อมูลต้นฉบับหรือบันทึก deviation หลังผู้ใช้ตัดสินใจ

## 2. การค้นหาต้นฉบับและ Dataset

- บทความระบุ dataset [InfantCry-DBL Version 1](https://data.mendeley.com/datasets/x493z8nmwc/1) DOI `10.17632/x493z8nmwc.1`; หน้าแหล่งข้อมูลระบุว่า dataset มี `metadata.csv` 1,551 แถว และระบุให้ยึดชื่อโฟลเดอร์แม่เป็น label สำหรับชุดที่เผยแพร่
- พบ [Version 2](https://data.mendeley.com/datasets/x493z8nmwc/2) แยกจาก Version 1 แต่ **ยังไม่ได้ตรวจไฟล์ต่อไฟล์หรือยืนยันว่าไฟล์ในเครื่องตรง version ใด** ชื่อโฟลเดอร์เดิมที่เคยตั้งในโครงการอื่นไม่ใช่หลักฐานว่า dataset เป็น V2
- จากการค้นเว็บด้วยชื่อบทความ/DOI และค้นชื่อ repository ผ่าน GitHub ณ วันที่ตรวจ **ยังไม่พบ source code ทางการที่ยืนยันได้** ไม่ใช่ข้อสรุปว่าไม่มี source code อยู่ที่อื่น
- บทความกล่าวว่าไฟล์เสียง online 324 clips มีข้อจำกัดการแจกจ่าย ขณะที่หน้า dataset มีคำอธิบายสิทธิ์ของชุดที่เผยแพร่ จึงต้องตรวจไฟล์และสิทธิ์รายแหล่ง/เวอร์ชันก่อนแจกจ่าย ไม่ได้ดาวน์โหลดหรือคัดลอกข้อมูลเพิ่ม
- [Mamba repository อย่างเป็นทางการ](https://github.com/state-spaces/mamba) เป็น implementation ทั่วไปของ Mamba **ไม่ใช่หลักฐานว่าเป็น source code ของ Sinc-Mamba-KAN ในบทความนี้** และไม่ควรเลือก release ล่าสุดอัตโนมัติแล้วอ้างว่าตรง paper

## 3. ผลตรวจข้อมูลที่มีอยู่ในเครื่อง — อ่านอย่างเดียว

วันที่ตรวจ `D:\INFANT CRY\data_set_dbl` มีไฟล์ WAV ในห้าโฟลเดอร์ รวม **1,551 ไฟล์** และมี `not_baby` อีก 2,000 ไฟล์แยกจากห้าคลาส บทความเป็น 5-class pre-cry classification จึงไม่นับ `not_baby` เป็นคลาสที่หก

| โฟลเดอร์ในเครื่อง | ชื่อ DBL ที่แมปเพื่อสำรวจ | จำนวน WAV |
|---|---|---:|
| belly_pain | eairh | 170 |
| burping | eh | 280 |
| discomfort | heh | 168 |
| hungry | neh | 217 |
| tired | owh | 716 |
| **รวม** | | **1,551** |

จำนวนรายคลาสข้างต้นเท่ากับผลบวกรายคลาส Dataset 1 + Dataset 2 ที่บทความรายงาน แต่ **จำนวนตรงกันไม่ยืนยันไฟล์ตรงกัน, label ถูกต้อง, แหล่งที่มา, version, tier หรือ infant ID** และโฟลเดอร์ในเครื่องเป็นโครงสร้างที่นำมาจัดใหม่ ไม่ใช่หลักฐานว่าเป็นโครงสร้างต้นฉบับของ dataset ที่เผยแพร่

การตรวจ SHA-256 แบบ byte-for-byte ของไฟล์ WAV ในห้าคลาสพบ **198 กลุ่มที่มีไฟล์ซ้ำ** ในจำนวนนี้ **3 กลุ่มมีไฟล์อยู่ต่างโฟลเดอร์คลาส** ตัวอย่างคู่ไฟล์:

| กลุ่ม | ไฟล์แรก | ไฟล์ที่สอง |
|---|---|---|
| 1 | `burping/dunstan-eh_48.wav` | `discomfort/4131_M_heh.wav` |
| 2 | `burping/dunstan-eh_5.wav` | `discomfort/2051_M_heh.wav` |
| 3 | `discomfort/3335_M_heh.wav` | `hungry/dunstan-neh_52.wav` |

ไฟล์ `burping/000543_M_heh.wav` มี suffix `heh` แต่โฟลเดอร์ที่จัดไว้แมปเป็น `eh` จึงถูกตั้งธงไว้ **ไม่สรุปอัตโนมัติว่าโฟลเดอร์หรือชื่อไฟล์ใดผิด** หน้าเผยแพร่ dataset ต้นฉบับระบุให้ใช้โฟลเดอร์แม่เป็น label แต่ต้องตรวจสอบ provenance และโครงสร้างต้นฉบับก่อนนำกฎนั้นมาสรุปกับสำเนาที่ถูกจัดใหม่ในเครื่อง

ไม่พบไฟล์ metadata/manifest `.csv` ภายใน root `data_set_dbl` ที่ตรวจ; ไม่ได้ค้นทุก directory บนเครื่อง จึงไม่อ้างว่า metadata ไม่มีอยู่ที่อื่น การตรวจ header ของ WAV ในห้าคลาสอ่านได้ครบ 1,551 ไฟล์โดยไม่มีข้อผิดพลาด รูปแบบที่พบรวม 16 kHz mono, 22.05 kHz mono และ 44.1 kHz stereo เป็นต้น

SHA-256 นี้เปรียบเทียบทั้งไฟล์แบบ byte-for-byte ไม่ตรวจคลิปที่ถูก resample, เปลี่ยน header, ถูกตัด หรือเสียงคล้ายกันแต่ bytes ต่างกัน และผลนี้ไม่ได้ยืนยันหรือหักล้าง overlap 180 clips ที่บทความรายงานระหว่าง dataset tiers

## 4. สภาพแวดล้อมที่ตรวจได้

- WSL Ubuntu ใช้ Python 3.10.12 และพบ `librosa 0.11.0` ใน environment `/home/adminuser/.venvs/audio-ml-gpu` ณ เวลาตรวจ
- `nvidia-smi` ใน WSL แสดง NVIDIA GeForce RTX 5070 Ti และ driver 621.14 ณ เวลาตรวจ เป็นเพียงการตรวจพบอุปกรณ์ ไม่ใช่หลักฐานว่า PyTorch/Mamba ใช้ GPU หรือ CUDA kernels ได้
- ใน environment เดิม `torch`, `mamba_ssm` และ `causal_conv1d` ไม่พบจากการตรวจ `importlib.util.find_spec`; จึงไม่ได้ทดสอบ forward pass หรือ GPU สำหรับ PyTorch
- ไม่ติดตั้ง/เปลี่ยน dependency ของ environment TensorFlow เดิม และยังไม่เสนอเวอร์ชัน PyTorch/Mamba แบบยืนยัน เพราะต้องตรวจรุ่น Python, CUDA, RTX 5070 Ti และ source paper ร่วมกันก่อนตาม [คำแนะนำ PyTorch](https://pytorch.org/get-started/locally/) และ [Mamba](https://github.com/state-spaces/mamba)

## 5. งานที่ทำได้แล้วและการทดสอบ

1. เก็บค่าที่บทความระบุได้ไว้ใน `configs/paper_full_capacity.json`; ค่าไม่ทราบเป็น `null`, `training_enabled` เป็น `false` และไฟล์นี้ไม่ได้สร้างโมเดลหรือเป็นคำสั่ง train
2. สร้าง `src/infant_benchmark/audit.py` สำหรับตรวจไฟล์ WAV แบบอ่านอย่างเดียว ไม่สร้าง split และไม่ระบุ infant ID
3. เพิ่ม automated tests ใน `tests/test_dataset_audit.py` ด้วยไฟล์ WAV จำลอง เพื่อทดสอบการนับ, suffix ที่ไม่ตรง, hash ซ้ำข้ามคลาส, hash ซ้ำภายในคลาส, root ที่ไม่มี และ WAV header เสีย รวมถึงตรวจว่าไฟล์ fixture ไม่ถูกแก้
4. **ยังไม่ได้** สร้าง SincConv/Mamba/KAN model, สกัด 213 features, fit scaler/MI, train, ประเมิน, สร้าง confusion matrix หรือเปรียบเทียบผลกับ `D:\INFANT CRY`

## 6. สิ่งที่ต้องขอผู้ใช้ก่อนทำส่วนที่ต้องเลือกสมมติฐาน

1. หากมี **source code ของผู้เขียน, supplementary material, config, split indices หรือวิธีสร้าง features 213 ตัว** กรุณาแนบ/ให้ลิงก์เพื่อยืนยัน exact implementation
2. หากมี **dataset ต้นฉบับ Version 1 พร้อม `metadata.csv`, README, tier และ infant/session IDs ที่พิสูจน์ความหมายได้** กรุณาระบุตำแหน่งและสิทธิ์ใช้งาน โดยยังไม่ย้ายหรือแก้ข้อมูลเดิม
3. หลังตรวจหลักฐานเพิ่มเติม หากรายละเอียดบางอย่างยังขาด ให้ผู้ใช้เลือกว่าจะ **รอข้อมูลจากผู้เขียน** หรืออนุมัติ **implementation แบบใกล้เคียงพร้อมบันทึก deviations** ซึ่งจะไม่เรียกว่า exact reproduction
4. กำหนด policy สำหรับไฟล์ suffix ไม่ตรง/ไฟล์ hash ซ้ำข้ามคลาสจากหลักฐานต้นฉบับ โดยไม่แก้ label อัตโนมัติ และกำหนดว่าข้อมูลใดใช้สำหรับ experiment ใด
5. ยืนยัน environment แยกสำหรับ PyTorch/Mamba และตรวจ forward pass ด้วยข้อมูลจำลองก่อนให้ผู้ใช้เป็นผู้เริ่ม train เอง

## 7. คำสั่งที่มีตอนนี้

ดูคำสั่งตรวจข้อมูลและรัน unit tests ได้ใน [README](../README.md) ยังไม่มีคำสั่ง train เนื่องจากตัวโมเดลและ protocol ยังไม่ผ่านเงื่อนไขข้างต้น
