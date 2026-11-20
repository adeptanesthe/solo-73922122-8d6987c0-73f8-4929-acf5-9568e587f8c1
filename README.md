# 冲压机冲次声音核对

值班员上传**正常循环参考录音**与**待检循环录音**，系统从音频中提取冲击并按参考节奏匹配，自动标出**漏冲对应的参考冲次**与**多冲的实录时刻**，网页可定位播放标出的时段。

同一网页另有**灵敏度试验**页：检修员换过麦克风位置后，用空转录音叠加单冲样音，试出「空转噪声下多弱的冲击还能被检出」，并给出通过的最低增益档。

## 输入约定

- WAV（PCM）：16 kHz、16-bit、单声道，单个文件不超过 8 秒
- 参考录音：含 4～8 次清晰冲击，相邻冲击间隔至少 150 ms
- 待检录音：允许起始静音、整体响度变化、漏冲、多冲；正常冲次间隔相对参考允许 ±15% 变化
- 灵敏度试验的空转录音：沿用同样的 WAV 格式；保持原音量
- 灵敏度试验的单冲样音：同样的 WAV 格式，时长不超过 80 ms，只含 1 次冲击

## 判定规则

1. **冲击检测**：短时能量包络 + 自适应门限（门限跟随录音自身峰值，对整体响度变化不敏感）+ 局部极大值拾取。
2. **回声处理**：同一冲击 40 ms 不应期内的峰（含衰减拖尾、回声）只保留最强一个，回声不计多冲。
3. **节奏匹配（动态规划）**：以参考冲次序号为基准做最优对齐，成本 = 漏冲数 + 多冲数；相邻匹配的累计时长须落在参考节奏 ±15% 窗口内。
4. **漏冲不错位**：漏冲只计一次成本，后续冲次仍按参考序号继续对齐，不会整体错位。
5. **歧义人工复听**：若多种归属具有相同的最优成本（同样满足节奏与最少漏/多冲），返回 `ambiguous`，网页给出醒目提示，不武断放行。
6. **无效音频不覆盖结果**：格式/时长/参考质量不合格时返回 400 且不写入状态，上一次有效结果原样保留。

## 灵敏度试验规则

1. **叠加方式**：空转录音保持原音量；单冲样音按「试验时刻 − 样音冲击时刻」的偏移
   放置，使样音里的冲击对准试验时刻，片段须完整落在空转录音内。
2. **试验时刻**：3～6 个，彼此至少相隔 150 ms；越界或片段放不下时返回 400。
3. **增益档**：样音增益从 −24 dB 到 0 dB 每 3 dB 一档（共 9 档），逐档独立重新混音。
4. **不可测**：混音峰值超出 16-bit PCM 量程（不可实际回放）的档位记为
   `unmeasurable`，不参与推荐。
5. **完整混音判定**：每个试验时刻前后 15 ms 内恰好检出 1 个峰，且窗外无峰。
6. **逐处撤除判定**：单独撤掉某处样音重新混音并检测，该处窗口内必须无峰。
7. **通过与推荐**：两条同时满足才算该档 `pass`；推荐通过的最低增益档
   （增益越低，能检出的冲击越弱）。无档通过时不给推荐，并逐档列出失败原因。
8. **输入有误撤下旧推荐**：试验输入不合格时返回 400，同时保存一条
   `status=input_error` 的记录，`recommendation` 为空并说明原因，网页不再展示旧推荐。

## 启动（Docker Compose）

```bash
docker compose up --build
# 浏览器打开 http://localhost:8000 ；顶部页签可切到「灵敏度试验」
```

录音与结果数据保存在命名卷 `press-data`（容器内 `/data`）。

## 本地开发（无 Docker 时）

```bash
python3 -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 自定义数据目录：APP_DATA_DIR=/some/dir uvicorn app.main:app
```

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/inspect` | multipart 表单字段 `reference`、`test`（均为 .wav），返回核对结果 |
| `GET` | `/api/result/last` | 最近一次有效结果（无结果时 404） |
| `DELETE` | `/api/result/last` | 清除内存中的最近结果 |
| `POST` | `/api/sensitivity` | multipart 表单字段 `idle`、`sample`、`impact_time`（秒）、`trial_times`（JSON 数组或逗号分隔），返回灵敏度试验结果 |
| `GET` | `/api/sensitivity/last` | 最近一次灵敏度试验记录（含输入有误的撤下记录；无记录时 404） |
| `DELETE` | `/api/sensitivity/last` | 清除灵敏度试验记录 |
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/data/uploads/{name}` | 回传已上传录音，供网页播放器定位播放 |

结果关键字段：`verdict`（`ok` / `defect` / `ambiguous`）、`missed[].ref_index`（漏冲对应的参考冲次，从 0 开始）、`missed[].expected_time`（待检中预计应出现的时刻）、`extras[].time`（多充实录时刻）、`matches`（匹配明细）、`warnings`（回声抑制、歧义等提示）。

灵敏度试验结果关键字段：`levels[]`（每档 `gain_db`、`status` = `pass` / `fail` /
`unmeasurable`、`mixed_peak`、`failure_reasons`、`full_mix`、`leave_one_out`）、
`recommendation`（`null` 或 `{gain_db, reason}`）、`recommendation_reason`（无推荐时说明原因）、
`summary`（通过/未通过/不可测档数）、`warnings`。

## 测试

```bash
python3 -m pytest tests/ -q
```

合成 WAV 生成器见 `scripts/gen_demo.py`，可用于手工演示：

```bash
python3 scripts/gen_demo.py /tmp/demo
```

除冲次核对的演示录音外，还会生成灵敏度试验用的 `idle.wav`（空转噪声）与
`sample.wav`（样音冲击位于 0.02s），网页上可填 `0.02` 与 `1.0, 2.0, 3.0, 4.0` 试用。

## 目录结构

```
app/
  audio.py       WAV 校验与冲击检测
  matching.py    节奏匹配动态规划（漏冲/多冲/歧义）
  sensitivity.py 灵敏度试验（样音叠加、增益扫描、撤除判定、推荐）
  service.py     检查服务与结果存储（无效输入不覆盖旧结果）
  main.py        FastAPI 路由
  static/        单页前端（冲次核对 + 灵敏度试验两个页签）
tests/           检测、匹配、灵敏度、端到端 API 测试（54 项）
scripts/         演示 WAV 生成
```
