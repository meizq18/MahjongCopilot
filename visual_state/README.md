# 纯视觉状态采集原型（2026-10-12）

这是独立入口，**不要启动原 main.py / BotManager / wrapper SDK**。
读取磁盘 PNG 或通过 mss 读取主屏幕；不建立代理、不读取 WebSocket、不点击或移动鼠标。
当前完成范围：手牌区域分割、可选复用旧 CNN、多帧稳定检查、结构化观察输出。
不是完整麻将状态机，也不能据此生成可信 Mortal 建议。

## 源码核对

基线（本地克隆，固定 commit）：
- MahjongCopilot: `31be3de972bbe4b1261360ff61b18f1bb84babbe`
- majsoul_wrapper: `d1667d527c3f8c44b37654deec8de46dad9d7f1f`

对应代码：
- [game_state.py](https://github.com/meizq18/MahjongCopilot/blob/31be3de972bbe4b1261360ff61b18f1bb84babbe/game/game_state.py)：`input/_input_inner` 分派 liqi 消息；`ms_auth_game` 确定账号与绝对座位；`ms_new_round` 创建 start_kyoku、初始手牌与 tsumo；后续动作转 MJAI。不能把截图手牌列表直接塞入这条消息链。
- [bot_manager.py](https://github.com/meizq18/MahjongCopilot/blob/31be3de972bbe4b1261360ff61b18f1bb84babbe/bot_manager.py)：`_run` 启动 MITM，`_process_msg` 解析 WebSocket；还调用自动操作。HUD 格式转换与 manager 耦合，因此原型不导入它。
- [action.py](https://github.com/meizq18/majsoul_wrapper/blob/d1667d527c3f8c44b37654deec8de46dad9d7f1f/action/action.py)：`_getHandTiles` 从 (235,1002) 扫描，白色阈值 110、floodFill、牌尺寸筛选；原版两帧取 min 消除高光。原型复用分割方法，增加界限/ROI/大区域排除，逐帧处理，避免两帧跨摸打动画时 min 混合不同牌。代价是高光下可能拒绝帧。
- [classifier.py](https://github.com/meizq18/majsoul_wrapper/blob/d1667d527c3f8c44b37654deec8de46dad9d7f1f/action/classifier.py)：38 类含红五与 back，两层 CNN；仅返回 argmax，没有置信度。通过文件路径只加载此模块，复用其 tile.model、BGR→RGB、32×32 与归一化，不加载 package init 或 GUIInterface。
- [README](https://github.com/meizq18/majsoul_wrapper/blob/d1667d527c3f8c44b37654deec8de46dad9d7f1f/README.md)：完整状态由 WebSocket SDK 输入，视觉属于动作输出侧。
- 本地模型 `bot/local/bot_local.py` 的 `BotMortalLocal` 及 `Bot.init_bot/react_batch` 可未来复用；HUD `game/browser.py` 的 `overlay_update_guidance/overlay_clear_guidance` 可复用接口。它们依赖的 libriichi、权重与浏览器运行环境未验证。

## 安装和运行

从 MahjongCopilot 仓库根目录执行（建议 Python 3.11/3.12 独立虚拟环境）：

```bash
python -m pip install -r visual_state/requirements.txt
python -m pytest visual_state/tests -q
# 首先只验证截图区域/分割；没有 classifier 时标签为 ?，永不 stable
python -m visual_state --frames ./screenshots --rect 0 0 1920 1080 --output observations.jsonl
# 安装本机匹配的 torch、torchvision、Pillow 后，复用可信 wrapper checkout
python -m visual_state --frames ./screenshots --rect 0 0 1920 1080 --wrapper ../majsoul_wrapper --output observations.jsonl
# 主屏幕只读采集，rect 相对主屏幕截图像素；100 帧后退出
python -m visual_state --live --rect 0 0 1920 1080 --wrapper ../majsoul_wrapper --interval 0.2 --count 100 --output observations.jsonl
```

手动选择游戏内容的 16:9 矩形，排除地址栏、窗口边框、黑边；不是整个浏览器窗口。
像素坐标按实际截图计，不按 Windows DPI 逻辑坐标。屏幕位置、缩放变化后重新校准。
仅支持与旧版 1920×1080 手牌布局对应的界面；没有自动 ORB 校准。裁剪区缩放至此分辨率。
文件按名字排序，使用 `000001.png` 形式。回放时间戳为合成值，不能用它证明无丢帧。
输出文件默认覆盖；每帧写一行 JSON。

`tile_boxes` 坐标属于归一化游戏图，不是桌面坐标；字段同时记录 frame_id/source。
未知 round/score/river/melds/seat/legal_actions 等均为 null；null 不等同空牌河、0 分或未立直。
1–14 张只检查基本检测范围，兼容副露后的较短手牌，不代表牌数/回合合法性已验证。
每帧当前手牌为原始观察；连续三帧标签和框完全一致才 stable。动画/unknown/不可能重复牌/采集间隔过长均撤销稳定。
相同牌含红五合计超过四张拒绝。稳定不证明分类正确，confidence=null。
没有把最右侧牌武断设为摸牌；drawn_tile 始终未知。

## 状态结构与扩展方案

`State` 包含 schema_version、帧与时间、手牌/框、稳定性/问题，以及局信息、座位映射、宝牌指示牌、摸牌、轮到谁、剩余牌、合法动作、四个 Player。
Player 使用 self/right/across/left 屏幕相对位置，未来必须另建绝对 actor 映射。
`Observed(value, source, frame_id, confidence)` 区分观察、推断与手动输入；后续推断应附原始帧、事件来源、歧义候选与时序范围。
当前 history_complete/model_ready 恒为 false，不会伪造已同步局面。

| 信息 | 单帧可见性与后续实现 | 为什么需要连续追踪 |
|---|---|---|
| 自己手牌 | 已分割；旧 CNN 接口已实现但真实分类未测 | 高光、抬牌、摸打动画和遮挡需要去抖 |
| 自己新摸牌、手切/摸切 | 静态牌间隔仅是线索 | 追踪牌位置、身份和操作前后；重复牌可仍有歧义 |
| 四家牌河 | 未来分区分类、旋转校正、网格定位 | 判定新增、撤走被鸣牌、立直横牌；不能只比较最终牌数 |
| 副露、吃碰明杠/加杠/暗杠 | 未来分类与横牌方向识别 | 判定发生顺序、来源 actor、消耗牌和杠类型 |
| 立直/立直接受 | 横牌、棒和分数是线索 | 宣言到扣分/接受是不同事件，不能合并 |
| 宝牌指示牌 | 未来分类可读取（不是宝牌本身） | 杠后新增时点及数量；不要仅输出最终列表 |
| 场风/局数/本场/供托/分数/规则 | 未来 OCR + 人工校准，均未实现 | 初局、下一局、结算界面切换及扣分 |
| 操作按钮/当前 actor/剩余牌 | 未来 OCR/模板与动画；有时不可见 | 显示时间短；按钮出现不能证明完整合法动作集合 |
| 绝对座位与初始状态 | 未来风位 OCR + 人工确认 | 从完整开局进入；中途截图不能恢复过去事件 |

无法可靠从普通对局截图得到：对手暗牌身份、牌山顺序、未翻开的里宝牌、服务器精确事件号/时间、没有显示的操作约束、被漏截或遮挡的历史。
对手摸牌可用 MJAI '?'，但摸牌发生顺序不能凭空补齐。对手是否摸切通常不能从最终牌河确定；连续动画也可能有歧义。
暗杠隐藏牌需依界面规则判断，不能臆造其红五组成。截图无法保证唯一还原所有服务器事件。

推荐后续顺序：真实手牌标注验证→牌河/副露/局面 OCR→完整开局录屏事件重建→MJAI 回放校验→本地 Mortal→只读 HUD。
事件重建器必须独立于观察器；确定才提交，歧义/丢帧时停止建议，等待人工核对或下局重置。
同一稳定截图不重复产生事件；新局清除候选与事件日志；Mortal 重建需从核验的 start_game/start_kyoku 回放，不能以最新手牌覆盖内部状态。

## MJAI / Mortal / HUD 接口

`mjai_tile` 已实现号码牌、字牌和红五转换；只转换牌编码，不转换牌局事件。
`MjaiBridge.events` 当前明确抛出未实现错误，即使调用者修改 ready 标志也不会送伪造事件。
未来输入应为带帧范围、来源、确认状态的有序事件；不能从手牌差分直接推定 tsumo/dahai。
`MortalPort.react_batch(events)` 与现有 Bot 对齐；实际接入前先 init_bot(verified_seat, GameMode.MJ4P)，核验完整开局、规则、事件顺序与 can_act 语义。初期限定四麻，三麻另行实现。
`HudPort` 对齐已有浏览器方法；`publish_observation` 先清旧提示再展示手牌及采集状态，无模型选项。只用 fake HUD 测过契约。
真实 HUD 还需拆出一个独立生命周期；不应借用 BotManager 启动路径，否则会启动 MITM/自动操作。

## 已验证与未验证

23 项 pytest 通过：合成白牌分割（1/7/13/14 张）、空帧、区域越界、缩放偏移、去抖、帧间隔、时间戳、红五重复、未知标签、MJAI 牌编码、fail-closed 桥、fake HUD、磁盘 PNG→CLI→JSONL、禁止网络/自动操作模块的直接导入，以及分类器加载不触发 GUI/package init。
测试中的 classifier 是假标签函数，**不是 CNN 准确率结果**。
未验证：真实牌局识别准确率、旧权重在新界面的效果、torch/torchvision 版本兼容、实际桌面 mss/DPI、帧率与漏动作率、Mortal 推理和实际 HUD。当前环境无 torch/torchvision、本地 Mortal 权重与桌面对局；wrapper 的 tile.model 已存在，但尚未运行 CNN。
固定阈值和布局可能在不同皮肤/亮度/缩放下失效；正确行为是 unknown/不稳定，而不是生成建议。

真实验证：
1. 录制可控回放并导出按顺序 PNG，标注每张手牌（保留红五）、框、摸打时点。至少覆盖开局、13/14张、副露后、立直、高光、遮挡、尺寸变化和断帧。
2. 人工核对每帧 JSON 手牌；分别统计单牌准确率、整手完全一致率、红五混淆、空帧误检、稳定帧误接受率和识别延迟。不要只报稳定帧精度而不报拒绝比例。
3. 桌面采集对比 PNG 回放结果；确认矩形像素、帧间隔和窗口移动后的失效行为。HUD 区域必须避开采集区域以免反馈污染。
4. 未来事件层用人工标注的有序 MJAI 真值比对每条事件；测试漏帧、鸣牌抢断、加杠、重复牌、新局。只有事件序列核对通过才能启用模型建议。

MIT 分割代码归属见 LICENSE.majsoul_wrapper；本目录随宿主仓库 GPL-3.0 分发，旧模型和分类器通过外部 checkout 复用，不捆绑复制权重。
