# CV Screening Bias Lab（履歷篩選偏見實驗室）：使用指南（中文）

**網上版本：** https://huggingface.co/spaces/edwardwww/cv-screening-bias-lab
**English version:** [USER_GUIDE_EN.md](USER_GUIDE_EN.md)

---

## 1. 這個應用程式是甚麼？

CV Screening Bias Lab 是一個互動教學示範，用來回答兩個問題：

1. **當履歷只改變了求職者的姓名、年齡、工作空檔或殘疾相關資料時，AI 履歷篩選工具會否給出不同的分數？**
2. **一個從未「看見」性別或年齡的招聘模型，仍然會歧視嗎？**

本程式是為香港城市大學 IS5113《人工智能倫理》小組專題（第一部分，題目一：*招聘平台的演算法偏見*）而製作，把近年的研究和法律案例，變成任何人都可以在瀏覽器中親手操作的實驗。

**所有履歷和數據均為虛構**，沒有使用任何真實人士的資料。

---

## 2. 背景：為甚麼這個問題重要？

越來越多僱主在真人閱讀履歷之前，先用 AI 為求職者排序、評分，甚至直接淘汰。近年的證據顯示，這些工具可能對條件相同的求職者作出不同對待：

| 來源 | 發現 |
|---|---|
| Wilson 與 Caliskan，華盛頓大學（AIES 2024） | 在 300 萬次履歷與職位配對中，AI 模型有 85% 的時間偏好白人姓名，而且從未偏好黑人男性姓名多於白人男性姓名。 |
| Bloomberg（2024） | GPT 會因姓名所暗示的種族和性別，對內容相同的履歷給出不同排名。 |
| Glazko 等（FAccT 2024） | 履歷若提及殘疾相關的獎項或活動，GPT-4 會把它排得較低。 |
| Rozado（2025） | 22 個大型語言模型都偏好女性姓名；更值得注意的是，它們傾向選擇**排在第一位**的候選人。 |
| *Mobley 訴 Workday*（美國，2023 年至今） | 集體訴訟指控一個 AI 篩選平台淘汰 40 歲以上、黑人及殘疾求職者，部分是透過「工作空檔」等代理變數。 |
| 紐約州審計長審計報告（2025 年 12 月） | 紐約市的 AI 招聘審計法例（Local Law 144）幾乎沒有被執行。 |

本程式讓你以小規模重現這類測試，親眼看到偏見是如何產生的。

---

## 3. 簡報示範（約 4 分鐘）：🎤 Presentation demo 分頁

第一個分頁用**四個步驟**講故事。先在頂部選擇職位（預設：香港 Marketing Executive）。

**Step 1 · 我們在測試甚麼？** 顯示香港職位廣告（「流利粵語及英語、良好中文書寫」的要求會以黃色標示）、標準履歷（黃色部分是唯一會改變的地方），以及 10 個姓名：香港、英國、非洲黑人、日本（各一男一女）、南亞及菲律賓。下方亦列出其他測試：年齡（2019 vs 1996 年畢業）、性別、工作空檔及殘疾。

**Step 2 · 同一份履歷，分數不同。** 由 AI 招聘員（大型語言模型 Qwen2.5-72B-Instruct）為每份履歷評分（0 至 100）並寫出簡短理由。選擇測試（🌏 姓名、🎂 年齡、⚧ 性別、👶 工作空檔、♿ 殘疾）：
- 柱狀條顯示每份履歷的分數（3 次平均），最高為綠色、最低為紅色；
- 下方方框顯示相差多少分，並以「一份明顯較弱的履歷」的分數作比較；
- 右邊並排顯示 AI 對**最高分**和**最低分**的理由；
- 💡 **Insight** 方框用簡單文字說明結果：會檢查 AI 的理由有沒有直接提到被改變的細節，還是用聽起來合理的說法掩蓋。如果差距屬於 AI 正常的隨機波動，也會如實說明。
- ⚧ 性別測試會顯示 5 位男性及 5 位女性的平均分，兩段理由來自同一背景、差距最大的一對姓名。

結果會被儲存，所以每次簡報都顯示相同數字，不用等模型。**🔄 Re-run live** 會重新詢問 AI（約 1 分鐘）並取代已儲存的結果。Step 2 需要 Space 密鑰 `HF_TOKEN`。

**Step 3 · AI 如何從歷史學到偏見。** 三張卡：(1) 過往招聘人員在技能相同的情況下，每 100 位男性聘用 55 位，女性只聘用 25 位；(2) 用這些決定訓練 AI，並刪除性別和年齡，但工作空檔和畢業年份仍然保留；(3) 對同樣合資格的女性和 40 歲以上人士，AI 給的分數低約 11 至 13 分。下方的 **「為甚麼 AI 會學到偏見？」** 用數據分三點解釋：AI 的「標準答案」就是過往的決定；每 100 位女性有 55 位有工作空檔，男性只有 5 位；過往有工作空檔的人每 100 位只有 27 位獲聘，沒有空檔的有 46 位，所以 AI 學會「有工作空檔 = 低分」。

**Step 4 · 用更好的提示修正偏見。** 文字框內是給 AI 招聘員的規則，例如「不要考慮國籍；粵語、英語和中文能力只按履歷所寫判斷；照顧家人的工作空檔不是弱點」，你可以自行修改。選擇測試後按 **▶ Apply instruction**，同一批履歷會在提示中加入你的規則後重新評分，頁面並排顯示 **Before（原本提示）** 和 **After（加入你的規則）**，包括分數差距及 AI 對最低分的理由。💡 Result 方框會說明明顯差距是否消失、仍然存在，或者原本就沒有。Marketing Executive 的預設規則已預先計算；修改規則或按 **🔄 Run again** 會即時詢問 AI（約 1 分鐘）。重點：更好的提示可以減少偏見，但不保證有效，所以修正後必須再次測試。

![Step 2](screenshot_step2.png)

![Step 3](screenshot_step3.png)

![Step 4](screenshot_step4.png)

### Step 3 的數據如何模擬
| 特徵 | 與工作相關？ | 說明 |
|---|---|---|
| `skill_test`、`years_experience`、`education_level` | 是 | 每組分佈相同。 |
| `career_gap`（工作空檔） | **否** | 55% 女性 vs 5% 男性曾有空檔（照顧家人），成為隱藏的**性別**訊號。 |
| `grad_year`（畢業年份） | **否** | 直接反映年齡，成為隱藏的**年齡**訊號。 |

- **真實資歷**只取決於與工作相關的特徵。
- **過往決定** = 真實資歷 −（偏見 × 是否女性）−（偏見 × 是否 40 歲以上）+ 隨機誤差。
- AI 為邏輯迴歸（logistic regression），以過往決定訓練，不包含性別和年齡，並用預留的 40% 測試數據評分。

---

## 4. 建議的簡報講稿（約 4 分鐘）

1. **Step 1**（30 秒）：「一份香港職位、一份標準履歷。我們複製多份，每份只改一個細節。公平的招聘員應該給每份相同分數。」
2. **Step 2，🌏 姓名**（45 秒）：「分數很接近，差距可能只是隨機波動。這裏沒有明顯偏見，這也是一個發現。」
3. **Step 2，👶 工作空檔**（45 秒）：「同一份履歷，兩年照顧家人的空檔就少了 5 分，而 AI 自己的理由亦提到空檔可能影響近期知識。」
4. **Step 3**（60 秒）：「即使刪除性別和年齡，用有偏見的過往決定訓練的 AI 仍會找到線索（工作空檔和畢業年份），並重複同樣的偏見。」
5. **Step 4，👶 工作空檔**（45 秒）：「我們在提示中加入公平規則，差距由 5.0 分跌至 3.3 分，不再是明顯差距。但提示不是保證，每次修正都要再測試。」

---

## 5. 查看及下載數據

示範用到的所有數據都是虛構的，並已公開為 Hugging Face 數據集：
**[edwardwww/cv-screening-bias-lab-data](https://huggingface.co/datasets/edwardwww/cv-screening-bias-lab-data)**。任何人都可以查看或下載，毋須登入。

| 子集（config） | 行數 | 內容 |
|---|---|---|
| `hiring_history` | 6,000 | Step 3：模擬的過往求職者及有偏見的人手招聘決定 |
| `cv_variants` | 80 | Step 1 及 Step 2：4 個職位廣告，以及只相差一個細節的履歷副本 |
| `llm_scores` | 177 | Step 2 及 Step 4：AI 招聘員已儲存的分數及理由（每份履歷 3 次） |

**在瀏覽器查看：** 打開上面的連結，頁頂的 **Dataset Viewer** 會顯示表格，可在 **Subset** 選擇想看的表。
可以排序、搜尋，亦可按 **Data Studio** 用 SQL 篩選。（剛上傳後，查看器可能暫時顯示「should be available soon」，稍後重新整理即可。）

**在瀏覽器下載：** 打開 **Files and versions** 分頁，進入 `data/`，按一個 CSV 檔（例如 `hiring_history.csv`），
再按 **download** 圖示。CSV 檔可以用 Excel 或 Google Sheets 打開。

**用 Python 讀取：**
```python
# pip install datasets pandas
from datasets import load_dataset

history = load_dataset("edwardwww/cv-screening-bias-lab-data", "hiring_history", split="train").to_pandas()
cvs     = load_dataset("edwardwww/cv-screening-bias-lab-data", "cv_variants",    split="train").to_pandas()
scores  = load_dataset("edwardwww/cv-screening-bias-lab-data", "llm_scores",     split="train").to_pandas()

print(history.groupby("gender")[["career_gap", "truly_qualified", "hired_historically"]].mean())
```
或者直接用 pandas 讀 CSV：
```python
import pandas as pd
url = "https://huggingface.co/datasets/edwardwww/cv-screening-bias-lab-data/resolve/main/data/hiring_history.csv"
history = pd.read_csv(url)
```
同一批檔案亦放在本 repository 的 [`dataset/`](../dataset) 資料夾，數據集說明
（[`dataset/README.md`](../dataset/README.md)）解釋每一欄的意思及數據如何生成（seed 7）。
應用程式本身不會讀取這個數據集：每次啟動時，它都會在記憶體中重新生成 Step 3 的數據。

---

## 6. 自行運行

**在自己的電腦上**
```
pip install -r requirements.txt
python app.py
```
打開 http://127.0.0.1:7860。Step 1 和 Step 3 毋須連線。Step 2 和 Step 4 會顯示 `demo_cache.json` 內已儲存的結果；如要重新詢問 AI，請把環境變數 `HF_TOKEN` 設為一個具有「Make calls to Inference Providers」權限的 Hugging Face 權杖（token）。

**在公開的 Space 上，** 已儲存的結果可免費查看；但重新詢問 AI（Re-run live、Run again、新的 Step 4 指示）需要擁有者密碼，因為每次都會用到擁有者的額度。

**在 Hugging Face Spaces 部署自己的版本：** 請參閱 [`HOW_TO_DEPLOY.md`](../HOW_TO_DEPLOY.md)。

### 檔案
| 檔案 | 用途 |
|---|---|
| `app.py` | Gradio 介面：🎤 Presentation demo 及 ℹ️ About and ethics |
| `demo3.py` | 四個步驟：頁面版面、AI 評分及理由、偏見判斷及提示修正 |
| `demo_cache.json` | 已儲存的 AI 分數，令每次簡報顯示相同數字 |
| `cv_data.py` | 虛構香港職位廣告、履歷範本及會改變的細節 |
| `variants.py` | 填寫履歷範本，令每份只相差一個細節 |
| `scorers.py` | 連接 AI 招聘員（預設 `Qwen/Qwen2.5-72B-Instruct`） |
| `history_sim.py` | Step 3：偏見歷史模擬及「看不見性別年齡」的 AI |

---

## 7. 限制與倫理

- **僅作示範，並非科學審計：** 履歷數量少、只有四個職位、一個模型、每份履歷評分 3 次。結果可能因職位、履歷、模型及每次運行而改變。「沒有偏見」的結果同樣是有價值的發現。
- **只使用虛構數據。** 姓名是香港勞動市場中各族群的常見名字，但不代表任何真實人士。
- **簡化的模擬：** Step 3 是解釋偏見如何被學習的模型，並非真實公司數據。
- **請把它當作審計工具**，不要用來尋找「得分最高」的名字，或優化履歷以騙過篩選系統。
- **AI 使用聲明：** AI 工具（Claude）協助編寫程式碼和文件；專題小組負責設計實驗、檢查及詮釋結果。

---

## 8. 參考資料

- Wilson, K. & Caliskan, A. (2024). *Gender, Race, and Intersectional Bias in Resume Screening via Language Model Retrieval*. AAAI/ACM AIES.
- Bloomberg (2024). *OpenAI's GPT Is a Recruiter's Dream Tool. Tests Show There's Racial Bias.*
- Glazko, K. et al. (2024). *Identifying and Improving Disability Bias in GPT-Based Resume Screening*. ACM FAccT.
- Rozado, D. (2025). *Gender and Positional Biases in LLM-Based Hiring Decisions*. arXiv:2505.17049.
- *Mobley v. Workday, Inc.*，美國加州北區聯邦地區法院（2025 年 5 月 16 日獲准以集體訴訟形式進行）。
- Office of the New York State Comptroller (2025). *Enforcement of Local Law 144: Automated Employment Decision Tools*.
- International Labour Organization (2018). *Care Work and Care Jobs for the Future of Decent Work*.
