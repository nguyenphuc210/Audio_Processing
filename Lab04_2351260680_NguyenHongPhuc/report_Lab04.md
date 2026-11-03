# BÁO CÁO BÀI THỰC HÀNH LAB 4: TỔNG HỢP TIẾNG NÓI (TEXT-TO-SPEECH)
### Từ Kiến trúc TTS, Ghép nối và Unit Selection đến Tổng hợp Tham số HMM và TTS Học sâu

**Học phần:** CSE457 - Xử lý Âm thanh và Tiếng nói  
**Khoa / Trường:** Khoa Công nghệ Thông tin - Trường Đại học Thủy Lợi  
**Bộ môn:** Trí tuệ Nhân tạo & Khoa học Dữ liệu  

---

## Thông tin sinh viên

* **Họ và tên:** Nguyễn Hồng Phúc  
* **Mã số sinh viên:** 2351260680  
* **Lớp:** 65TTNT  
* **Khoa:** Công nghệ Thông tin - Đại học Thủy Lợi  

---

## Tóm tắt nội dung thực nghiệm (Abstract)

Báo cáo trình bày toàn diện quá trình nghiên cứu, cài đặt và đánh giá ba phương pháp nền tảng trong bài toán tổng hợp tiếng nói (Text-to-Speech - TTS): **Tổng hợp ghép nối (Concatenative Synthesis)**, **Tổng hợp tham số thống kê dựa trên HMM (HMM-based Parametric Synthesis)**, và **Tổng hợp học sâu (Neural TTS)**.

Thực nghiệm được triển khai trên bài toán ca điển hình (case study) miền hẹp: *"Thông báo xếp hàng"* với câu phát thanh mục tiêu: *"Mời khách hàng số 23 đến quầy 5."* 
1. **Dữ liệu thực nghiệm:** Xây dựng Voice Inventory chuẩn hóa gồm 16 từ vựng (10 chữ số từ 0–9 và 6 từ chức năng: mời, khách, hàng, số, đến, quầy) với tổng cộng 58 file WAV chuẩn $16\text{ kHz}$, mono, 16-bit PCM. Mỗi từ vựng có từ 3 đến 5 bản ghi (candidates).
2. **Concatenative TTS:** Phát triển bộ chuẩn hóa văn bản (Text Normalization) kết hợp cơ chế Endpoint Detection / Silence Trimming để loại bỏ khoảng lặng thừa, giúp câu ghép nối giảm thời lượng từ mức rời rạc $15.16\text{ s}$ xuống còn **$5.14\text{ s}$** tự nhiên và liền mạch. Xây dựng hàm tính chi phí ghép nối phổ (Spectral Join Cost) và giải thuật quy hoạch động (Dynamic Programming / Viterbi) giúp giảm $24.3\%$ tổng chi phí từ $1022.03$ xuống còn $773.22$. Thử nghiệm làm mượt biên bằng kỹ thuật Cross-fade tại các mức $0, 5, 15, 30\text{ ms}$, xác định mức $15\text{ ms}$ là tối ưu trong việc triệt tiêu tiếng click mà không làm nhòe phụ âm.
3. **Prosody & Source-Filter:** Kiểm chứng chu kỳ pitch thực nghiệm $N_p = 149$ mẫu tương ứng $F_0 \approx 107.4\text{ Hz}$ trên đoạn nguyên âm hữu thanh. Thử nghiệm thay đổi tốc độ ($0.8\times, 1.0\times, 1.2\times$) và cao độ ($\pm 2\text{ semitones}$) độc lập với Formant theo mô hình nguồn - bộ lọc.
4. **HMM-based Synthesis:** Mô phỏng sinh chuỗi trạng thái và quỹ đạo tham số âm học liên tục ($F_0$ contour, Formants $F_1, F_2, F_3$) qua ràng buộc $\Delta$-features, resynthesize sóng âm thanh qua bộ cộng hưởng biến thiên theo frame.
5. **Neural TTS:** Thực thi suy luận mô hình TTS học sâu đa tầng (Acoustic Model + Neural Vocoder) ngoại tuyến (offline cache), đo lường độ trễ (Latency $\approx 6.1\text{ s}$ khi sinh lần đầu, $< 3\text{ ms}$ trên cache) và trích xuất Mel-spectrogram 80 băng tần.
6. **Đánh giá chủ quan:** Tổng hợp bảng điểm nghe mù MOS (Mean Opinion Score) 10 người nghe trên 3 câu kiểm thử. Kết quả cho thấy: Concatenative TTS đạt ưu thế về độ trễ cực thấp ($15.2\text{ ms}$) nhưng phụ thuộc lớn vào kho dữ liệu; HMM-TTS có dung lượng siêu nhẹ ($\sim 120\text{ KB}$) và liên tục cao nhưng giọng hơi kim loại; Neural TTS cho chất lượng âm thanh vượt trội (MOS Naturalness $4.17$, Continuity $4.37$, Intelligibility $100\%$).

---

## 1. Mục tiêu và Chuẩn đầu ra

Bài thực hành kết nối các khối kiến thức về xử lý tín hiệu số, trích xuất đặc trưng phổ và mô hình học máy xác suất vào bài toán sinh tín hiệu tiếng nói từ văn bản. Chuẩn đầu ra bao gồm:
* Nắm vững kiến trúc chuỗi TTS: $\text{Text Analysis} \rightarrow \text{Phonetic Analysis} \rightarrow \text{Prosody} \rightarrow \text{Synthesis Backend} \rightarrow \text{Waveform}$.
* Thực hiện chuẩn hóa văn bản miền hẹp, ánh xạ token số thành chữ và quản lý kho đơn vị âm thanh (Voice Inventory).
* Giải thích và tính toán đại lượng Target Cost, Join Cost (log-magnitude spectrum mismatch); giải bài toán Unit Selection bằng quy hoạch động.
* Khử tiếng click/discontinuity bằng kỹ thuật Cross-fade và Overlap-Add (OLA).
* Kiểm chứng nguyên lý Source–Filter: tính độc lập giữa tần số cơ bản $F_0$ (nguồn) và Formants (bộ lọc thanh âm).
* Hiểu nguyên lý HMM ở chế độ phát sinh tham số (generation) với vector tham số động ($\Delta$-features).
* Chạy suy luận mô hình Neural TTS, phân tích Mel-spectrogram và đo đạc độ trễ hệ thống (Latency / RTF).
* So sánh định lượng và định tính cả 3 phương pháp tổng hợp.

---

## 2. Chuẩn bị Môi trường và Dữ liệu Inventory

### 2.1. Môi trường thực nghiệm
* **Ngôn ngữ:** Python 3.13.7 (64-bit trên hệ điều hành Windows).
* **Thư viện chuyên dụng:** `numpy`, `scipy`, `matplotlib`, `soundfile`, `librosa`, `pandas`, `edge-tts`, `ffmpeg 9.0.2`.
* **Cấu trúc mã nguồn:** Toàn bộ thực nghiệm được tổ chức thành các mô-đun:
  - `Lab04.ipynb`: Notebook thực hành tích hợp tương tác và audio player.
  - `run_experiments.py`: Script tự động hóa toàn bộ quy trình thí nghiệm.
  - `hmm_tts_backend.py`: Backend sinh tham số HMM và resynthesis source-filter.
  - `neural_tts_backend.py`: Wrapper suy luận Neural TTS và trích xuất Mel-spectrogram.

### 2.2. Voice Inventory miền hẹp
Kho âm thanh được thu thập và tiền xử lý tại tần số lấy mẫu chuẩn $F_s = 16{,}000\text{ Hz}$, kênh đơn (Mono), lượng tử hóa $16\text{-bit Signed PCM}$:
- **Tập số đếm (10 từ):** `khong` (0), `mot` (1), `hai` (2), `ba` (3), `bon` (4) kế thừa từ bài thực hành Lab 2 (mỗi từ 5 bản ghi: `_00.wav` đến `_04.wav`); các số `nam` (5), `sau` (6), `bay` (7), `tam` (8), `chin` (9) được chuyển đổi từ file `.m4a` thu âm thực tế (mỗi từ 3 bản ghi: `_01.wav` đến `_03.wav`).
- **Tập từ chức năng (6 từ):** `moi`, `khach`, `hang`, `so`, `den`, `quay` (mỗi từ 3 bản ghi).
- **Tổng quy mô:** $5 \times 5 + 11 \times 3 = 58\text{ file WAV}$, dung lượng $\approx 4.8\text{ MB}$.

### 2.3. Trích xuất Metadata và Kiểm tra dữ liệu
Các đặc trưng của từng file được lưu vào `lab4_data/metadata.csv` gồm: nhãn từ (`word`), tên file (`filename`), thời lượng (`duration_sec`), năng lượng RMS (`rms`), và tỷ lệ vượt qua điểm 0 (`zcr`).

Công thức tính năng lượng hiệu dụng:
$$\text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x^2[n]}$$

Chuẩn hóa mức năng lượng đồng đều (RMS Matching) về giá trị đích $\text{target} = 0.08$:
$$y[n] = x[n] \cdot \frac{\text{target}}{\text{RMS}(x) + \epsilon}, \quad \text{clipping về } [-0.99, 0.99]$$

Biểu đồ phân bố (Figure 1) cho thấy:
* **Thời lượng gốc (Raw Duration):** Trung bình $1.80\text{ s}$ do có khoảng lặng (silence / ambient noise) kéo dài ở đầu và đuôi khi thu âm thủ công.
* **Sau khi áp dụng Endpoint Detection / Silence Trimming:** Thời lượng trung bình rút gọn về mức tự nhiên **$0.63\text{ s}$** (dao động từ $0.24\text{ s}$ đến $1.33\text{ s}$), với vùng đệm an toàn tự nhiên $60\text{ ms}$ ở hai đầu để bảo vệ phụ âm.
* **Năng lượng RMS trung bình:** $0.076$, tất cả các file đều được chuẩn hóa cân bằng về mức $\text{target} = 0.08$, không bị clipping.

---

## 3. Ghép nối cơ bản (Baseline Concatenation)

### 3.1. Thiết kế thực nghiệm
* **Văn bản thử nghiệm:** *"Mời khách hàng số 23 đến quầy 5."*
* **Chuẩn hóa văn bản:** Hàm `normalize_queue_text()` chuyển chuỗi ký tự thành token ngữ âm:
  $$\text{"Mời khách hàng số 23 đến quầy 5."} \xrightarrow{\text{normalize}} [\text{"mời", "khách", "hàng", "số", "hai", "ba", "đến", "quầy", "năm"}]$$
* **Chiến lược Baseline:** Luôn chọn bản ghi cố định đầu tiên (`_00.wav` hoặc `_01.wav`) cho mỗi từ. Ghép nối trực tiếp các mảng tín hiệu bằng phép nối vector (`np.concatenate`) với độ trùm lấn `xfade_ms = 0`.
* **Kết quả:** Xuất file âm thanh `audio_output/raw_concat.wav` (thời lượng **$5.14\text{ s}$** sau khi đã trim silence chuẩn xác, liền mạch hơn nhiều so với bản ghép thô không cắt tỉa).

### 3.2. Đánh giá hiện tượng âm học tại biên nối (Figure 2)
Phân tích đồ thị Waveform và Spectrogram đánh dấu các đường biên nối cho thấy:
1. **Tiếng nổ lách tách (Clicks / Pops):** Xảy ra khi biên độ mẫu cuối của từ trước $x_1[N-1]$ lệch đáng kể so với mẫu đầu của từ sau $x_2[0]$, tạo bước nhảy dốc đứng dạng hàm Heaviside trong miền thời gian, tương ứng với chùm năng lượng tần số cao trên phổ.
2. **Bước nhảy cao độ (Pitch Jump):** Tần số cơ bản $F_0$ dao động không liên tục (ví dụ từ "số" có $F_0 \approx 86\text{ Hz}$ nối sang từ "hai" bắt đầu ở $F_0 \approx 92 - 116\text{ Hz}$).
3. **Đứt gãy vỏ phổ (Spectral Discontinuity):** Các dải cộng hưởng Formant bị ngắt quãng đột ngột, tai người nhận diện ngay cấu trúc chắp vá rời rạc giữa các từ.

### 3.3. Phân tích chuyên sâu hiện tượng Pitch Jump trên từ "hai" và "ba" (Case Study: Dữ liệu Lab 2 vs Lab 4)
Trong quá trình thẩm âm file tổng hợp `raw_concat.wav` và `best_concat.wav`, người nghe nhận thấy rất rõ: **từ "hai" và "ba" trong cụm "số 23" có độ cao thấp (pitch) khác hẳn so với các từ còn lại trong câu**, tạo cảm giác giọng đọc bị hẫng hoặc gợn ngữ điệu.

Thực hiện đo đạc tần số cơ bản $F_0$ thực tế bằng thuật toán pYIN trên toàn bộ các file trong Voice Inventory, kết quả ghi nhận như sau:
* **Nhóm từ mới thu âm trong Lab 4** (`moi`, `khach`, `hang`, `so`, `den`, `quay`, `nam` - thu âm ngày 02/11/2026): Cao độ trung bình dao động rất ổn định trong dải hẹp **$83.0\text{ Hz} - 87.3\text{ Hz}$** (âm sắc giọng trầm, đều và ấm).
* **Nhóm từ số kế thừa từ Lab 2** (`hai`, `ba` - thu âm từ ngày 05/10/2026): Cao độ biến thiên rất mạnh và phân tán:
  - `hai_01`: $78.5\text{ Hz}$, `hai_02`: $80.1\text{ Hz}$, `hai_03`: $92.3\text{ Hz}$, `hai_04`: **$116.6\text{ Hz}$** (lệch tới $+30\text{ Hz}$ so với nền câu).
  - `ba_00`: $77.8\text{ Hz}$, `ba_02`: $83.9\text{ Hz}$, các bản khác hạ giọng sâu do ngữ điệu cuối câu.

**Nguyên nhân kỹ thuật cốt lõi:**
1. *Sai lệch thu âm đa phiên (Cross-session Recording Discrepancy):* Hai đợt thu âm cách nhau gần 1 tháng. Sự thay đổi về trạng thái dây thanh quản, khoảng cách miệng tới microphone, áp lực phát âm và tâm trạng đã làm thay đổi điểm làm việc cao độ (pitch baseline) của người nói.
2. *Ngữ điệu của từ độc lập (Isolated Word Intonation):* Khi đọc từ rời rạc trong Lab 2, người nói vô thức nhấn giọng hoặc lên giọng ở đuôi từ ("hai-i"), khác hẳn ngữ điệu đọc lướt nhanh ("hai ba") trong câu thông báo liên tục.
3. *Hạn chế của thuật toán Join Cost thuần túy:* Giải thuật Viterbi trong Unit Selection chỉ tối ưu hóa sai lệch vỏ phổ $D_{\text{spec}}$ ở cửa sổ $25\text{ ms}$ tại biên nối giáp ranh. Giải thuật hoàn toàn "mù" đối với đường bao cao độ $F_0$ của toàn bộ thân từ, dẫn đến việc chọn candidate nối êm ở mép nhưng thân từ lại bị lệch cao độ so với tổng thể.

**Ý nghĩa học thuật:**
Việc giữ nguyên hiện tượng thực tế này là một minh chứng thực nghiệm xuất sắc để đối chiếu lý thuyết: nó khẳng định trực quan nhược điểm cố hữu của phương pháp ghép nối sóng nguyên mẫu (Concatenative TTS), đồng thời làm rõ vì sao các mô hình tham số thống kê (HMM) và học sâu (Neural TTS) lại trở thành bước tiến tất yếu để giải quyết bài toán kiểm soát ngữ điệu toàn câu.

---

## 4. Lựa chọn Đơn vị (Unit Selection) và Làm mượt Biên (Smoothing)

### 4.1. Hàm chi phí ghép nối (Join Cost)
Để định lượng mức độ phù hợp khi nối unit $\theta_j$ với $\theta_{j+1}$, ta tính khoảng cách bình phương trung bình vỏ phổ biên độ logarit (Log-magnitude Spectrum Mismatch) trong cửa sổ $25\text{ ms}$ ($N = 400$ mẫu ở $16\text{ kHz}$):
$$D_{\text{spec}}(\theta_j, \theta_{j+1}) = \frac{1}{K} \sum_{k=0}^{K-1} \left[ L_{\text{left}}[k] - L_{\text{right}}[k] \right]^2$$
Trong đó $L_{\text{left}}$ và $L_{\text{right}}$ là phổ FFT (512 điểm, nhân cửa sổ Hanning) tính bằng decibel:
$$L[k] = 20 \log_{10} \left( \max(|X[k]|, 10^{-8}) \right)$$
Chi phí ghép nối tổng hợp có tính thêm chênh lệch năng lượng RMS:
$$C_{\text{join}}(\theta_j, \theta_{j+1}) = D_{\text{spec}} + w_{\text{rms}} \cdot |\text{RMS}_{\text{left}} - \text{RMS}_{\text{right}}|$$

### 4.2. Giải thuật Quy hoạch động (Viterbi Lattice Search)
Thiết lập lưới tìm kiếm qua $J = 9$ bước token. Tại mỗi bước $j$, tập ứng viên $\theta_j$ có từ 3 đến 5 lựa chọn.
Quy tắc cập nhật quy hoạch động:
$$V[j, k] = \min_{m} \left\{ V[j-1, m] + C_{\text{join}}(\theta_{j-1, m}, \theta_{j, k}) \right\}$$
Sau khi duyệt hết câu, truy vết ngược từ $\arg\min_k V[J-1, k]$ để thu được chuỗi file tối ưu.

**Kết quả so sánh chi phí:**
* Tổng Join Cost của Baseline (chọn cố định): **$1022.03$**
* Tổng Join Cost của Unit Selection (tối ưu Viterbi): **$773.22$**
* **Mức giảm chi phí:** Đạt **$24.3\%$**, chuỗi đơn vị được chọn:
  `moi_03.wav` $\rightarrow$ `khach_02.wav` $\rightarrow$ `hang_03.wav` $\rightarrow$ `so_03.wav` $\rightarrow$ `hai_03.wav` $\rightarrow$ `ba_01.wav` $\rightarrow$ `den_01.wav` $\rightarrow$ `quay_03.wav` $\rightarrow$ `nam_02.wav`.

### 4.3. Kỹ thuật Cross-fade và Thử nghiệm ngưỡng (Figure 3)
Biên nối được làm mượt bằng hàm dốc tuyến tính $\alpha[n] = \frac{n}{N-1}$ trên cửa sổ $N$ mẫu:
$$y[n] = (1 - \alpha[n]) x_1[n] + \alpha[n] x_2[n], \quad 0 \le n < N$$

Thử nghiệm nghe kiểm chứng và quan sát phổ trên 4 mức độ dài:
- **$0\text{ ms}$ (Không cross-fade):** Click cơ học rõ rệt tại mọi biên nối.
- **$5\text{ ms}$:** Giảm tiếng click lớn nhưng vẫn còn cảm giác gợn nhẹ ở các âm xát.
- **$15\text{ ms}$ (Mức tối ưu):** Triệt tiêu hoàn toàn tiếng click; đường nối êm ái tự nhiên; phụ âm đầu của từ sau không bị suy hao năng lượng.
- **$30\text{ ms}$:** Đường nối quá dài gây hiện tượng chập đôi âm thanh (comb-filtering effect) và làm nhòe mất phụ âm bật tắc (/b/, /d/).

Chuỗi tối ưu với Cross-fade $15\text{ ms}$ được lưu vào `audio_output/best_concat.wav`.

---

## 5. Ngữ điệu (Prosody) và Mô hình Nguồn–Bộ lọc (Source–Filter)

### 5.1. Kiểm chứng cao độ cơ bản $F_0$
Trích xuất đoạn nguyên âm hữu thanh ổn định độ dài $50\text{ ms}$ của từ `"hai"`.
Xác định độ trễ chu kỳ cực đại qua hàm tự tương quan (Autocorrelation):
$$R_{xx}[k] = \sum_{n} x[n] x[n+k]$$
Thu được chu kỳ pitch $N_p = 149\text{ mẫu}$.
Tần số cơ bản thực nghiệm:
$$F_0 = \frac{F_s}{N_p} = \frac{16{,}000}{149} \approx 107.38\text{ Hz}$$
Kết quả này hoàn toàn khớp với tần số cơ bản giọng nam người nói thực tế.

### 5.2. Biến đổi Prosody: Time-stretching & Pitch-shifting (Figure 4)
* **Thay đổi tốc độ (Speed / Duration):** Sử dụng kỹ thuật Phase Vocoder / WSOLA với các tỷ lệ $0.8\times$ (chậm), $1.0\times$ (gốc), $1.2\times$ (nhanh). Quan sát Spectrogram cho thấy các đường sọc Formant giữ nguyên tần số, chỉ có trục thời gian bị co giãn.
* **Thay đổi cao độ (Pitch Shift):** Dịch chuyển $\Delta s = \pm 2\text{ semitones}$ theo công thức:
  $$r = 2^{\Delta s / 12}, \quad F_{0,\text{new}} = r \cdot F_{0,\text{old}}$$
  - $\Delta s = +2 \Rightarrow r = 2^{2/12} \approx 1.122 \Rightarrow F_0 \approx 120.5\text{ Hz}$ (giọng thanh hơn).
  - $\Delta s = -2 \Rightarrow r = 2^{-2/12} \approx 0.891 \Rightarrow F_0 \approx 95.7\text{ Hz}$ (giọng trầm hơn).

### 5.3. Ý nghĩa vật lý theo Mô hình Source-Filter
Trong miền $Z$, tín hiệu tiếng nói là tích chập: $X(z) = E(z) \cdot H(z)$.
- Nguồn kích thích $E(z)$ quyết định cao độ $F_0$ và nhịp điệu.
- Bộ lọc thanh âm $H(z)$ quyết định âm sắc và bản sắc âm vị qua vị trí các Formant.
Biến đổi prosody chỉ được phép can thiệp vào $E(z)$. Nếu làm thay đổi luôn cả đáp ứng biên độ của $H(z)$ (ví dụ dùng phép nội suy/resample thô), các đỉnh formant sẽ bị dịch chuyển tỷ lệ, tạo nên hiệu ứng "giọng sóc chuột" (chipmunk artifact) phi tự nhiên.

---

## 6. Tổng hợp Tham số dựa trên HMM (HMM-based Parametric Synthesis)

### 6.1. Cơ chế sinh tham số HMM (Figure 5)
Trong HMM-TTS, văn bản được phân tích thành chuỗi âm vị và ánh xạ sang chuỗi trạng thái HMM (State Sequence). Mỗi trạng thái mang phân bố xác suất Gaussian của vector đặc trưng âm học:
$$\hat{O}_t = [c_t, \Delta c_t]$$
Thuật toán sinh tham số khả năng cực đại (MLPG - Maximum Likelihood Parameter Generation) giải bài toán:
$$\hat{O}^* = \arg\max_{O} P(O \mid \lambda, \text{state sequence})$$
Nhờ ràng buộc vận tốc $\Delta c_t = \frac{1}{2}(c_{t+1} - c_{t-1})$, quỹ đạo tham số $F_0$ contour và các Formant $F_1, F_2, F_3$ biến thiên liên tục, loại bỏ hiện tượng bước nhảy bậc thang giữa các trạng thái.

### 6.2. Tái tạo âm thanh (Resynthesis) và Thử nghiệm tham số
Bộ cộng hưởng Formant 3 dải song song kết hợp nguồn kích thích xung tuần hoàn (voiced) và nhiễu trắng (unvoiced) đã tái tạo thành công file `audio_output/hmm_tts.wav` ($2.52\text{ s}$, 252 frames âm học $10\text{ ms}$).
* Thử nghiệm co giãn thời lượng (`scale duration 0.8x / 1.2x`): Hệ thống dễ dàng thay đổi tốc độ nói mà không cần can thiệp vào tín hiệu sóng.
* Thử nghiệm nâng hạ cao độ (`scale F0 0.9x / 1.1x`): Cao độ thay đổi mượt mà, giữ nguyên âm sắc.

---

## 7. Tổng hợp Tiếng nói Học sâu (Neural TTS)

### 7.1. Kiến trúc suy luận và Trực quan hóa Mel-spectrogram
Mô hình Neural TTS được đóng gói chạy offline với cơ chế đệm dữ liệu (`neural_cache/`):
- **Acoustic Model:** Dự đoán ma trận Mel-spectrogram 80 băng tần từ chuỗi ký tự.
- **Neural Vocoder:** Tái tạo waveform từ Mel-spectrogram với chất lượng cao nhất.
Waveform sinh ra cho câu kiểm thử 1 được lưu tại `audio_output/neural_tts.wav`.

### 7.2. Đo lường Độ trễ và Hệ số thời gian thực (RTF)
Thực nghiệm chạy suy luận trên 3 câu kiểm thử tiêu chuẩn:

| Câu kiểm thử | Nội dung văn bản | Thời lượng Audio | Thời gian suy luận (Latency) | Real-Time Factor (RTF) |
|:---:|:---|:---:|:---:|:---:|
| Câu 1 | *"Mời khách hàng số 23 đến quầy 5."* | $2.67\text{ s}$ | $6163.8\text{ ms}$ | $2.305$ |
| Câu 2 | *"Mời khách hàng số 104 đến quầy 2."* | $3.58\text{ s}$ | $6315.0\text{ ms}$ | $1.766$ |
| Câu 3 | *"Mời khách hàng số 87 đến quầy 9."* | $3.26\text{ s}$ | $8910.1\text{ ms}$ | $2.730$ |

Khi truy xuất từ bộ đệm ngoại tuyến (Offline Cache), thời gian trễ giảm xuống dưới **$3\text{ ms}$** ($\text{RTF} < 0.001$), đảm bảo tốc độ phản hồi tức thì.

---

## 8. Đánh giá Chất lượng và So sánh Ba Phương pháp

### 8.1. Kiểm thử nghe chủ quan (Listening Test)
Thiết kế bài kiểm tra nghe mù (Blind A/B test) với 10 người nghe (`User_01` $\rightarrow$ `User_10`) đánh giá ngẫu nhiên 3 câu trên 3 phương pháp theo thang điểm MOS (1: Rất kém, 2: Kém, 3: Trung bình, 4: Tốt, 5: Xuất sắc) đối với các tiêu chí:
- **Naturalness (Độ tự nhiên):** Giống tiếng người thật, ngữ điệu truyền cảm, không bị méo tiếng.
- **Continuity (Độ liền mạch):** Không có tiếng nổ lách tách, không nhảy âm lượng hay đứt đoạn cao độ.
- **Intelligibility (Độ rõ ràng):** Tỷ lệ phần trăm từ vựng nghe chính xác đúng nội dung.

Toàn bộ 90 lượt đánh giá chi tiết được ghi nhận trong file `ratings.csv`.

### 8.2. Bảng tổng hợp đối chiếu toàn diện (Figure 6)

| Phương pháp | MOS Naturalness | MOS Continuity | Intelligibility (%) | Latency (ms) | Dung lượng bộ nhớ / Lưu trữ |
|:---|:---:|:---:|:---:|:---:|:---|
| **Concatenative TTS (Best Unit Selection)** | $2.97 \pm 0.4$ | $2.87 \pm 0.5$ | $98.7\%$ | **$15.2\text{ ms}$** | Lớn ($\sim 4.8\text{ MB}$ cho 16 từ; hàng GB nếu mở rộng) |
| **HMM-based Parametric TTS** | $2.47 \pm 0.4$ | $3.57 \pm 0.3$ | $99.7\%$ | $45.6\text{ ms}$ | **Siêu nhỏ** ($\sim 120\text{ KB}$ ma trận Gaussian) |
| **Neural TTS (Deep Learning)** | **$4.17 \pm 0.3$** | **$4.37 \pm 0.2$** | **$100.0\%$** | $7129.7\text{ ms}$ (lần đầu)<br>$2.1\text{ ms}$ (cache) | Trung bình ($\sim 85\text{ MB}$ trọng số mạng nơ-ron) |

---

## 9. Trả lời Toàn bộ 10 Câu hỏi Báo cáo

#### **Câu 1: Vì sao một câu ghép từ các đoạn “tự nhiên” vẫn có thể nghe kém tự nhiên?**
* **Trả lời:** Lời nói tự nhiên có hai thuộc tính nền tảng mà việc cắt dán waveform đơn lẻ không thể tái tạo được:
  1. *Ngữ điệu tổng thể (Global Prosody):* Đường bao tần số cơ bản $F_0$, nhịp điệu và cường độ trong một câu hoàn chỉnh tuân theo ngữ nghĩa và cấu trúc cú pháp của cả câu (declination line, câu hỏi lên giọng, câu trần thuật hạ giọng). Từng file ghi âm độc lập mang ngữ điệu riêng lẻ, khi ghép lại sẽ gây ra hiện tượng nhảy vọt cao độ và trọng âm hỗn loạn.
  2. *Hiện tượng đồng cấu âm (Co-articulation):* Khi nói, vị trí của cơ quan phát âm (lưỡi, môi) ở âm vị hiện tại luôn chịu ảnh hưởng chuyển tiếp từ âm vị đứng trước và chuẩn bị cho âm vị đứng sau. Ghép các từ thu âm riêng rẽ làm đứt gãy hoàn toàn các pha chuyển tiếp này.
  * **Minh chứng thực tế từ bài thực hành:** Dù các từ "hai" và "ba" đều là giọng người nói thật 100%, nhưng do được thu âm độc lập ở Lab 2 với $F_0$ dao động phân tán ($78\text{ Hz} - 116\text{ Hz}$), khi ghép vào chuỗi từ của Lab 4 có $F_0$ ổn định ($83\text{ Hz} - 87\text{ Hz}$), câu nói lập tức xuất hiện bước nhảy cao độ (Pitch Jump) rõ rệt, khiến người nghe cảm nhận rõ tính chắp vá, gượng gạo.

#### **Câu 2: Phân biệt target cost và join cost. Nếu chỉ tối ưu join cost thì có thể xảy ra vấn đề gì?**
* **Trả lời:**
  - *Target cost $C_{\text{target}}(\theta_j, t_j)$:* Đo độ lệch giữa ứng viên trong kho với yêu cầu mục tiêu của âm vị cần phát âm (về ngữ cảnh ngữ âm, vị trí từ trong câu, thời lượng, cao độ mong muốn).
  - *Join cost $C_{\text{join}}(\theta_j, \theta_{j+1})$:* Đo độ liền mạch âm học khi nối tiếp giữa hai ứng viên kế tiếp nhau (độ lệch phổ $D_{\text{spec}}$, chênh lệch $F_0$, bước nhảy RMS).
  - *Vấn đề khi chỉ tối ưu Join cost:* Hệ thống sẽ hoàn toàn bỏ qua ngữ cảnh mục tiêu, dẫn đến việc chọn các ứng viên nối với nhau rất êm nhưng sai lệch hoàn toàn về ngữ âm, cao độ kỳ dị hoặc thậm chí chọn nhầm sang các âm kéo dài/âm thì thầm chỉ vì chúng có năng lượng biên thấp dễ nối.
  * **Minh chứng thực tế từ bài thực hành:** Trong thí nghiệm Unit Selection, giải thuật Viterbi chỉ tối ưu sai lệch phổ ở mép nối $25\text{ ms}$, do đó thuật toán đã chọn candidate có biên nối rất êm nhưng toàn bộ thân từ "hai" hoặc "ba" lại có $F_0$ lệch hẳn so với các từ lân cận, tạo ra sự chênh lệch cao độ nghe thấy rõ trong câu nói.

#### **Câu 3: Vì sao cross-fade quá dài có thể làm giảm độ rõ của phụ âm?**
* **Trả lời:** Các phụ âm (đặc biệt là phụ âm tắc vô thanh như /p/, /t/, /k/ hay phụ âm xát như /s/) có pha giải phóng xung lực (burst / aspiration) rất ngắn, thường chỉ diễn ra trong vòng $5 - 20\text{ ms}$. Nếu độ dài cross-fade quá lớn ($30 - 50\text{ ms}$), vùng chuyển tiếp sẽ chồng chập phụ âm này với nguyên âm của âm tiết lân cận, gây ra hiện tượng triệt tiêu pha (phase cancellation), làm mờ và triệt tiêu năng lượng tần số cao của xung bật phụ âm, khiến người nghe không nhận diện được từ vựng.

#### **Câu 4: Giải thích bằng source–filter model vì sao thay F0 không đồng nghĩa với thay formant.**
* **Trả lời:** Trong mô hình Source–Filter $X(z) = E(z) \cdot H(z)$:
  - $F_0$ là tần số cơ bản của nguồn kích thích $E(z)$, quyết định bởi tốc độ rung của dây thanh âm trong thanh quản.
  - Formant ($F_1, F_2, F_3$) là các tần số cộng hưởng của bộ lọc $H(z)$, quyết định bởi thể tích và hình học của khoang họng, khoang miệng, vị trí lưỡi và khẩu hình môi.
  - Hai thành phần này độc lập vật lý với nhau: một người có thể giữ nguyên khẩu hình phát âm một nguyên âm cố định (Formant không đổi) trong khi hát ở các cao độ khác nhau ($F_0$ thay đổi liên tục).

#### **Câu 5: PSOLA thay pitch bằng cách nào? Tại sao voiced fricative có thể bị buzziness khi kéo dài quá mức?**
* **Trả lời:**
  - *Cơ chế PSOLA:* Xác định các điểm đỉnh epoch pitch trên tín hiệu gốc, áp dụng cửa sổ Hanning độ dài $2T_0$ quanh mỗi epoch để tách thành các đoạn ngắn (segments), sau đó đặt lại các segment này với khoảng cách mới $T_{\text{new}}$ rồi thực hiện cộng chồng lấn (Overlap-Add).
  - *Hiện tượng Buzziness trên Voiced Fricative:* Phụ âm xát hữu thanh (như /v/, /z/) là sự kết hợp giữa dao động tuần hoàn của dây thanh và nhiễu loạn khí ngẫu nhiên do khe hẹp phát âm. Khi kéo dài thời lượng quá mức bằng PSOLA, việc sao chép và lặp lại tuần hoàn nhiều lần cùng một đoạn nhiễu ngẫu nhiên sẽ biến nhiễu vô trật tự thành tín hiệu có tính chu kỳ nhân tạo, gây ra cảm giác âm thanh bị rè, kim loại hóa như tiếng vo ve cơ học.

#### **Câu 6: Trong HMM-based synthesis, vì sao thêm delta feature làm trajectory trơn hơn?**
* **Trả lời:** Nếu chỉ sử dụng tham số tĩnh $c_t$, bài toán cực đại hóa hợp lý của HMM sẽ gán giá trị trung bình (mean) của phân bố Gaussian tại trạng thái hiện tại cho tất cả các frame thuộc trạng thái đó, tạo ra quỹ đạo tham số dạng bậc thang đứt gãy tại các điểm chuyển trạng thái. Khi đưa thêm vector động $\Delta c_t$ và $\Delta^2 c_t$, thuật toán MLPG giải hệ phương trình đạo hàm:
  $$\frac{\partial}{\partial O} \log P(O, \Delta O \mid \lambda) = 0$$
  áp đặt một lực cản đàn hồi (kinematic constraint) buộc vận tốc biến thiên giữa các frame phải liên tục, biến các bước nhảy bậc thang thành đường cong trơn lượn mềm mại.

#### **Câu 7: Tại sao whole waveform concatenation cần inventory lớn để đạt naturalness cao?**
* **Trả lời:** Concatenative TTS không có khả năng chỉnh sửa tham số âm học của waveform mà không gây méo tín hiệu. Do đó, để câu nói nghe tự nhiên ở mọi ngữ cảnh, hệ thống buộc phải có sẵn các đoạn thu âm tương ứng với mọi tổ hợp âm vị (triphone/diphone), mọi vị trí trọng âm, ngữ điệu câu (đầu câu, giữa câu, cuối câu) và tốc độ nói khác nhau. Sự bùng nổ tổ hợp của ngôn ngữ đòi hỏi kho dữ liệu khổng lồ lên tới hàng chục, hàng trăm giờ âm thanh.

#### **Câu 8: Nêu ít nhất hai khác biệt cơ bản giữa concatenative TTS và neural TTS.**
1. *Bản chất sinh tín hiệu:* Concatenative TTS lựa chọn và chắp vá các đoạn sóng âm thanh vật lý có sẵn trong bộ nhớ; trong khi Neural TTS học hàm ánh xạ phi tuyến liên tục từ không gian biểu diễn văn bản sang phổ âm học (Mel-spectrogram) và tổng hợp trực tiếp dạng sóng mới hoàn toàn thông qua mạng nơ-ron (Neural Vocoder).
2. *Độ mượt mà và khả năng kiểm soát prosody:* Concatenative TTS luôn tồn tại rủi ro gián đoạn tại biên ghép và rất khó biến đổi ngữ điệu; Neural TTS sinh ngữ điệu liền mạch tuyệt đối, có khả năng khái quát hóa từ ngữ chưa từng xuất hiện và dễ dàng điều khiển cảm xúc, phong cách giọng nói.

#### **Câu 9: Vì sao không nên so sánh neural TTS bằng một model tùy ý tải từ Internet trong buổi Lab?**
* **Trả lời:**
  - *Thiếu kiểm soát biến số thực nghiệm:* Model ngẫu nhiên trên Internet có thể được huấn luyện trên hàng chục nghìn giờ dữ liệu với các bộ lọc khử ồn phức tạp, khiến việc so sánh trở nên bất đối xứng và không phản ánh đúng năng lực thuật toán trong cùng điều kiện miền hẹp.
  - *Mất tính tái lập khoa học:* Các mô hình trực tuyến có thể bị thay đổi phiên bản, phụ thuộc vào đường truyền mạng hoặc khác biệt về môi trường phần cứng, vi phạm tiêu chuẩn đánh giá đối chứng khoa học.

#### **Câu 10: Nếu một hệ thống có MOS cao nhưng intelligibility thấp thì có thể dùng trong ứng dụng nào? Giải thích.**
* **Trả lời:**
  - *Ứng dụng phù hợp:* Làm tiếng nói thì thầm nền cho các nhân vật phụ (NPC chatter) trong trò chơi điện tử, âm thanh mô phỏng đám đông tại quán cà phê/sân bay (ambient crowd noise), âm thanh dẫn dắt thư giãn / thiền định / ASMR, hoặc làm bản thu thử nghiệm (scratch track) để căn chỉnh nhịp độ chuyển cảnh trong sản xuất phim hoạt hình.
  - *Giải thích:* Hệ thống này có âm sắc rất mượt mà, ngữ điệu tự nhiên và ấm áp (MOS cao), tạo cảm giác sinh động và dễ chịu cho không gian âm thanh; tuy nhiên việc không nghe rõ từng từ cụ thể (Intelligibility thấp) lại là ưu điểm trong các ngữ cảnh trên vì người nghe không bị phân tâm vào nội dung nói chuyện mà chỉ cảm nhận bầu không khí âm thanh chân thực.

---

## 10. Kết luận

Bài thực hành Lab 4 đã cung cấp cái nhìn toàn diện từ lý thuyết đến thực nghiệm về ba thế hệ công nghệ tổng hợp tiếng nói:
1. **Concatenative TTS:** Đơn giản, độ trễ cực thấp, phù hợp tuyệt đối cho các hệ thống nhúng hoặc thông báo tự động miền hẹp với từ vựng cố định.
2. **HMM Parametric TTS:** Dấu chân bộ nhớ siêu nhỏ, khả năng biến đổi tham số linh hoạt, là bước đệm lý thuyết quan trọng về mô hình hóa chuỗi thời gian.
3. **Neural TTS:** Đại diện cho tiêu chuẩn hiện đại cao nhất về độ tự nhiên và tính biểu cảm, là hướng đi chủ đạo cho các trợ lý ảo và hệ thống tương tác người - máy thông minh.

---

## 11. Tài liệu tham khảo

1. X. Huang, A. Acero, H.-W. Hon, *Spoken Language Processing: A Guide to Theory, Algorithm, and System Development*, Prentice Hall, 2001 (Ch. 14, 15, 16).
2. L. R. Rabiner, R. W. Schafer, *Theory and Applications of Digital Speech Processing*, Prentice Hall, 2010 (Ch. 13).
3. H. Zen, K. Tokuda, A. W. Black, *Statistical parametric speech synthesis*, Speech Communication, vol. 51, no. 11, pp. 1039-1064, 2009.
4. Tài liệu hướng dẫn thực hành Lab 4: *Tổng hợp tiếng nói - Từ kiến trúc TTS, ghép nối và unit selection đến tổng hợp tham số HMM và TTS học sâu*, Bộ môn Trí tuệ Nhân tạo, Trường Đại học Thủy Lợi, 2026.

