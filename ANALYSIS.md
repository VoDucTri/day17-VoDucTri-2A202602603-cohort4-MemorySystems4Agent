# BÁO CÁO PHÂN TÍCH HỆ THỐNG BỘ NHỚ (MEMORY SYSTEMS FOR AI AGENT)
**Cohort 4 - Day 17: Memory Systems for AI Agent**

---

## 1. Kết Quả Thực Nghiệm Benchmark

Dưới đây là kết quả thực nghiệm thu được khi chạy lệnh:
```bash
python src/benchmark.py
```

### 1.1. Bảng 1: Standard Benchmark (`data/conversations.json` - 10 hội thoại, user `dungct`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 3,987 | 26,601 | **0.0%** | 10.0% | 0 | 0 |
| **Advanced** | 4,291 | 35,252 | **100.0%** | **100.0%** | 311 | 0 |

### 1.2. Bảng 2: Long-Context Stress Benchmark (`data/advanced_long_context.json` - 16 lượt dài, user `dungct_stress`)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 659 | 31,100 | **0.0%** | 10.0% | 0 | 0 |
| **Advanced** | 1,279 | **11,716** | **100.0%** | **100.0%** | 233 | **11** |

---

## 2. Phân Tích Chuyên Sâu Các Trade-Off

### 2.1. Vì sao Advanced Agent có Recall vượt trội so với Baseline Agent (100% vs 0%)?
- **Baseline Agent (Within-session memory only)**: Trạng thái hội thoại (`SessionState`) được lưu theo khóa duy nhất là `thread_id`. Khi một câu hỏi recall được gửi trong một phiên (thread) hoàn toàn mới, `BaselineAgent` không có bất kỳ dữ liệu nào về người dùng từ các phiên trước. Vì vậy, khả năng nhớ chéo phiên (**Cross-session recall**) của Baseline luôn là **0%**.
- **Advanced Agent (3-layer memory architecture)**:
  1. *Short-term memory*: Bộ đệm tin nhắn gần nhất trong thread.
  2. *Persistent memory (`User.md`)*: Các thực thể bền vững (tên, nơi ở, nghề nghiệp, đồ uống yêu thích, sở thích, thú cưng, phong cách trả lời) được trích xuất và lưu trữ bền vững trên đĩa tại `state/profiles/<user_id>/User.md`.
  3. Khi bước sang thread mới, `AdvancedAgent` nạp thông tin từ `User.md` vào prompt context, cho phép agent trả lời chính xác tất cả các câu hỏi kiểm tra thông tin cá nhân dù ở bất kỳ phiên làm việc nào.

### 2.2. Vì sao Advanced Agent tốn nhiều Prompt Token hơn ở hội thoại ngắn (Standard Benchmark)?
- Ở bảng Standard Benchmark (10 hội thoại ngắn):
  - Baseline tiêu thụ **26,601** prompt tokens.
  - Advanced tiêu thụ **35,252** prompt tokens (cao hơn khoảng 32%).
- **Nguyên nhân**:
  - Với mỗi lượt chat, `AdvancedAgent` luôn đính kèm tệp hồ sơ người dùng `User.md` vào context để đảm bảo agent nắm được bối cảnh người dùng.
  - Trong các hội thoại ngắn (dưới ngưỡng compact `compact_threshold_tokens = 800`), cơ chế nén (`Compactions = 0`) chưa được kích hoạt vì tổng số token trong phiên chưa đủ lớn. Do đó, việc chèn thêm `User.md` tạo ra một chi phí phụ tải cố định (*fixed overhead*) trên từng turn.
  - **Trade-off kết luận**: Ở tác vụ ngắn, persistent memory đánh đổi một lượng token prompt phụ để lấy 100% độ chính xác nhớ thông tin người dùng.

### 2.3. Vì sao Compact Memory giúp Advanced Agent chiến thắng áp đảo ở hội thoại dài (Stress Benchmark)?
- Ở bảng Long-Context Stress Benchmark (16 lượt với ngữ cảnh kỹ thuật, tin tức dài hàng nghìn từ):
  - Baseline tiêu thụ tới **31,100** prompt tokens.
  - Advanced chỉ tiêu thụ **11,716** prompt tokens (giảm tới **62.3%** chi phí prompt!).
  - Số lần compaction kích hoạt: **11 lần**.
- **Nguyên nhân**:
  - `BaselineAgent` duy trì toàn bộ lịch sử thô qua từng turn. Lượng context dồn tích tăng theo cấp số cộng khiến tổng số prompt token được xử lý tăng theo cấp số nhân ($O(N^2)$).
  - `AdvancedAgent` tích hợp `CompactMemoryManager`. Mỗi khi tổng token trong thread vượt quá ngưỡng 800 tokens, các tin nhắn cũ sẽ được nén thành bản tóm tắt súc tích (`summarize_messages`), chỉ giữ nguyên văn `keep_messages = 4` tin nhắn gần nhất.
  - Cơ chế này chặn đứng sự bùng nổ ngữ cảnh, chuyển độ phức tạp prompt token tích lũy từ $O(N^2)$ về dạng tuyến tính $O(N)$.
  - Nhờ vậy, `AdvancedAgent` vừa tiết kiệm hơn 62% chi phí xử lý ngữ cảnh, vừa giữ trọn vẹn 100% recall nhờ lớp persistent memory `User.md`.

### 2.4. Phân tích độ tăng trưởng Memory File (Memory Growth) và các Rủi Ro Tiềm Ẩn
- **Tốc độ tăng trưởng**:
  - Standard Benchmark: tệp `User.md` tăng từ 0 lên 311 bytes.
  - Stress Benchmark: tệp `User.md` tăng từ 0 lên 233 bytes.
- **Rủi ro đi kèm trong môi trường production**:
  1. *Phình to vô hạn (Unbounded Growth)*: Nếu người dùng trò chuyện hàng trăm phiên và mọi chi tiết vụn vặt đều bị lưu, `User.md` sẽ phình to, làm tăng chi phí token khi nạp vào system prompt và có thể vượt quá context window.
  2. *Lưu trữ sai lệch do nhiễu (Noise Pollution)*: Nếu không có bộ lọc câu đùa, câu giả định hay thông tin công tác ngắn ngày (như trường hợp "Hà Nội chỉ đi họp 2 ngày" hay câu đùa "chuyển sang làm product manager"), hồ sơ sẽ bị ô nhiễm bởi các dữ liệu không chuẩn xác.
  3. *Mâu thuẫn thông tin (Fact Collision)*: Khi người dùng đổi nghề (từ backend sang MLOps) hoặc chuyển nơi ở (từ Đà Nẵng sang Huế), nếu hệ thống chỉ append mà không ghi đè (replace/update), agent sẽ giữ đồng thời cả hai thông tin mâu thuẫn.

---

## 3. Các Tính Năng Bonus Đã Triển Khai (Mục tiêu 90-100 Điểm)

1. **Conflict Handling & Fact Correction (Xử lý mâu thuẫn và đính chính)**:
   - Trong `extract_profile_updates` và `UserProfileStore.upsert_facts`, hệ thống tự động nhận diện các mẫu câu đính chính: `"giờ mình đang ở Huế chứ không còn ở Đà Nẵng"`, `"mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer"`.
   - Fact mới sẽ ghi đè và thay thế hoàn toàn fact cũ sai trong `User.md`, đảm bảo agent không bao giờ trả lời lẫn lộn nghề cũ hoặc địa chỉ cũ.

2. **Noise Rejection & Intent Filtering (Lọc nhiễu và từ chối lưu sai)**:
   - Phát hiện các mẫu câu đùa: `"chỉ là câu đùa"`, `"đùa với đồng nghiệp"` -> từ chối lưu `product manager`.
   - Phát hiện thông tin tạm thời: `"chỉ là nơi mình vừa bay ra họp"`, `"ví dụ cũ"` -> từ chối lưu `Hà Nội` hay `Đà Nẵng` cũ làm nơi ở.
   - Bỏ qua các câu hỏi thông thường kết thúc bằng dấu `?` để tránh hiểu nhầm câu hỏi truy vấn là lời khai báo thông tin cá nhân.
   - Tách biệt tên thú cưng (`corgi tên Bơ`) với tên người dùng (`DũngCT`), ngăn ngừa lỗi gán nhầm tên người thành tên chó.

3. **Cumulative Structured Entity Extraction (Trích xuất thực thể tích lũy có cấu trúc)**:
   - Các trường thông tin được chuẩn hóa theo format markdown key-value: `- **Tên**: ...`, `- **Nơi ở**: ...`, `- **Nghề nghiệp**: ...`, `- **Mối quan tâm chính**: ...`.
   - Với các sở thích kỹ thuật (`tech_interests`), hệ thống tự động tích lũy hợp nhất (union) các mối quan tâm qua từng phiên (`Python, AI, MLOps`) thay vì ghi đè mất mát.

---

## 4. Hướng Dẫn Kiểm Thử và Tái Lập Kết Quả

Từ thư mục gốc của repository:

1. **Chạy benchmark:**
   ```bash
   python src/benchmark.py
   ```
2. **Chạy bộ kiểm thử tự động:**
   ```bash
   pytest src/test_agents.py -v
   ```
   *Kết quả: 4/4 test passed xanh (read/write/edit User.md, compact trigger, cross-session recall, prompt load reduction).*
