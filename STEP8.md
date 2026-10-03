# BƯỚC 8: PHÂN TÍCH KẾT QUẢ BENCHMARK (STEP8.md)

Tài liệu này trả lời chi tiết 4 câu hỏi trọng tâm của **Bước 8** trong `Guide.md` dựa trên dữ liệu thực nghiệm đo đạc từ hệ thống.

---

## Bảng Kết Quả Benchmark Đối Chứng

### Standard Benchmark (`data/conversations.json` - 10 hội thoại)
| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 3,987 | 26,601 | 0.0% | 10.0% | 0 | 0 |
| **Advanced** | 4,291 | 35,252 | 100.0% | 100.0% | 311 | 0 |

### Long-Context Stress Benchmark (`data/advanced_long_context.json` - 16 lượt dài)
| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 659 | 31,100 | 0.0% | 10.0% | 0 | 0 |
| **Advanced** | 1,279 | 11,716 | 100.0% | 100.0% | 233 | 11 |

---

## Trả Lời 4 Câu Hỏi Của Bước 8

### 1. Vì sao Advanced có recall tốt hơn Baseline?
- **Số liệu chứng minh:** 
  - Cross-session recall của **Baseline** ở cả hai benchmark là **0.0%**.
  - Cross-session recall của **Advanced** ở cả hai benchmark đạt **100.0%**.
- **Cơ chế kỹ thuật:**
  - `BaselineAgent` chỉ quản lý session theo `thread_id` tạm thời trong RAM. Khi chuyển sang một thread mới để hỏi câu hỏi recall, `BaselineAgent` không còn bất kỳ dữ liệu nào về các phiên trước và phản hồi chưa có thông tin.
  - `AdvancedAgent` có lớp **Persistent Memory (`UserProfileStore`)**. Trong quá trình hội thoại, hàm `extract_profile_updates()` trích xuất các fact cốt lõi (tên, nghề nghiệp, nơi ở, phong cách, sở thích) và lưu trữ bền vững vào tệp đĩa `state/profiles/<user_id>/User.md`. Khi mở thread mới, `AdvancedAgent` nạp nội dung `User.md` vào prompt context, cho phép trả lời chính xác tất cả các câu hỏi kiểm tra chéo phiên.

### 2. Vì sao Advanced có thể tốn hơn Baseline ở hội thoại ngắn?
- **Số liệu chứng minh:**
  - Ở Standard Benchmark, `Prompt tokens processed` của Advanced là **35,252**, cao hơn Baseline (**26,601** tokens - chênh lệch khoảng 32%).
- **Cơ chế kỹ thuật:**
  - Trong mỗi lượt chat, `AdvancedAgent` luôn đính kèm hồ sơ `User.md` vào context để duy trì nhận thức về người dùng.
  - Trong các hội thoại ngắn (khoảng 10 lượt), tổng số token chưa vượt qua ngưỡng `compact_threshold_tokens = 800`, do đó lớp nén không kích hoạt (`Compactions = 0`). Chi phí đính kèm `User.md` trở thành phụ phí cố định (*overhead*) cho từng turn mà chưa được bù đắp bởi việc nén.
  - Đây là sự đánh đổi (*trade-off*): chấp nhận phụ phí token nhỏ ở hội thoại ngắn để đổi lấy 100% khả năng nhớ dài hạn.

### 3. Vì sao Compact Memory giúp Advanced có lợi thế ở hội thoại dài?
- **Số liệu chứng minh:**
  - Ở Long-Context Stress Benchmark, `Prompt tokens processed` của Advanced giảm từ **31,100 xuống 11,716 tokens** (tiết kiệm **62.3%** chi phí ngữ cảnh).
  - Cột `Compactions` của Advanced ghi nhận **11 lần nén**.
- **Cơ chế kỹ thuật:**
  - Ở hội thoại dài gồm nhiều đoạn văn bản lớn, `BaselineAgent` kéo theo toàn bộ lịch sử thô qua từng turn, khiến lượng prompt token tích lũy tăng phi mã theo cấp số nhân ($O(N^2)$).
  - `AdvancedAgent` trang bị `CompactMemoryManager`. Khi kích thước ngữ cảnh vượt ngưỡng 800 tokens, hệ thống tự động gọi `summarize_messages()` để cô đọng các tin nhắn cũ thành bản tóm tắt ngắn và chỉ giữ nguyên văn `keep_messages = 4` tin nhắn gần nhất.
  - Cơ chế này chặn đứng sự bùng nổ ngữ cảnh, chuyển độ phức tạp về dạng tuyến tính $O(N)$ và tối ưu hóa vượt bậc ở cột `Prompt tokens processed`.

### 4. File memory tăng trưởng ra sao và rủi ro gì đi kèm?
- **Số liệu chứng minh:**
  - `Memory growth (bytes)` của Advanced là **311 bytes** (Standard) và **233 bytes** (Stress), tương ứng với kích thước tệp `User.md` được tạo mới trên đĩa. Baseline luôn là 0 byte vì không lưu trữ đĩa.
- **Rủi ro đi kèm trong thực tế:**
  1. *Nguy cơ phình to (Unbounded Growth):* Nếu người dùng trò chuyện dài ngày và hệ thống lưu trữ thiếu chọn lọc, `User.md` sẽ tăng kích thước liên tục, làm tăng token phụ tải trên mỗi request và có thể tràn context window của LLM.
  2. *Nhiễu dữ liệu (Noise & Hallucination):* Người dùng có thể nói đùa (ví dụ: đùa làm product manager) hoặc nhắc tới địa điểm tạm thời (đi họp Hà Nội). Nếu không có bộ lọc nhiễu (*noise rejection*), các thông tin sai sẽ bị lưu vĩnh viễn vào hồ sơ.
  3. *Mâu thuẫn thông tin (Fact Inconsistency):* Khi thông tin thay đổi (chuyển từ Đà Nẵng sang Huế, đổi từ backend sang MLOps), nếu hệ thống chỉ append mà không có cơ chế *conflict handling / update*, `User.md` sẽ chứa cả thông tin cũ lẫn mới, gây sai lệch khi recall.
