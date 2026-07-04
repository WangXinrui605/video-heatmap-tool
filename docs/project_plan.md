# Video Heatmap Tool for Gaze Data - 项目计划与任务拆解

## 1. 项目范围确认

根据导师补充说明：

> Video heatmap only works for the txt file.
>
> In the DemoData.zip, ignore the json file for now.

因此，当前项目范围应该明确为：

> 使用 `Demo Data.zip` 中 `Face-to-Face` 数据集里的 `.txt` gaze data 和对应视频，开发一个 video heatmap 工具。

导师补充说明中也提到：实现不一定必须使用 Python，但 Python 是 preferred。因此当前技术路线仍建议使用 Python，因为它适合快速处理表格数据、视频帧和图像叠加。

当前阶段先忽略：

- `Augmented Reality` 文件夹中的 JSON 数据；
- AR/3D gaze ray 数据；
- 3D avatar heatmap；
- Unity 工具复刻；
- 自己训练 object detector；
- detector 算法比较；
- 大规模用户研究。

JSON 数据可以在报告中作为“future work”或“out of current scope”提到，但不应该作为第一版实现目标。

## 2. 项目目标

### 2.1 总体目标

开发一个工具，用于读取眼动追踪 `.txt` 数据，将 gaze 坐标映射到对应视频画面上，并生成可配置的视频热力图，帮助研究人员分析参与者在交互实验中看向哪里。实现语言优先选择 Python。

### 2.2 英文项目描述

可以在报告或 tracker 中使用：

> This project develops a Python video heatmap tool for TXT-based eye-tracking recordings in the provided Face-to-Face dataset. The tool reads gaze coordinates from the TXT file, maps them onto the corresponding scene video, generates configurable heatmap overlays, exports visual results, and evaluates correctness, usability, and processing performance.

### 2.3 具体目标

项目应完成以下目标：

1. 读取 `Face-to-Face` 数据中的 `.txt` gaze file。
2. 读取对应 participant 的 scene camera video。
3. 从 `.txt` 文件中提取 gaze x/y 坐标。
4. 过滤无效 gaze 数据。
5. 在视频帧上显示 gaze point，用于验证坐标是否正确。
6. 基于 gaze point 生成 heatmap。
7. 将 heatmap 半透明叠加到视频上。
8. 支持基本参数配置。
9. 导出两个最终结果：包含全部数据的静态 heatmap 图片，以及展示 heatmap 随时间发展的动态视频。
10. 对工具做简短 evaluation。

## 3. 数据结构理解

### 3.1 数据来源

当前主要数据来自：

```text
Demo Data.zip
└── Sui Li Demo Data
    ├── Augmented Reality        # 当前先忽略
    └── Face-to-Face             # 当前重点
        ├── P2
        ├── P3
        ├── P4
        └── P5
```

当前只使用 `Face-to-Face` 文件夹。

### 3.2 每个 participant 的数据

每个 participant 文件夹通常包含：

```text
P2/
├── ..._CsvData.txt
├── ..._Dikablis Glasses 3_Scene Cam.mp4
└── ..._Dikablis Glasses 3_Scene Cam_Original_Eye Tracking Video.mp4
```

可以这样理解：

- `.txt` 文件：眼动数据表格，包含 gaze 坐标和其他 eye-tracking 信息。
- `Scene Cam.mp4`：场景摄像头视频，适合作为 heatmap overlay 的基础视频。
- `Original_Eye Tracking Video.mp4`：可能已经包含 gaze red dot，适合用来验证坐标是否读对。

### 3.3 `.txt` 文件格式

`.txt` 文件虽然扩展名是 txt，但内容是表格数据，字段之间通常用 tab 分隔。

重要字段包括：

- `rec_time`：记录时间。
- `UTC`：时间戳。
- `Dikablis Glasses 3_Field Data_Scene Cam_Original_Gaze_Gaze X`
- `Dikablis Glasses 3_Field Data_Scene Cam_Original_Gaze_Gaze Y`

当前最可能使用的 gaze 坐标列是：

```text
Dikablis Glasses 3_Field Data_Scene Cam_Original_Gaze_Gaze X
Dikablis Glasses 3_Field Data_Scene Cam_Original_Gaze_Gaze Y
```

这两个字段应该表示 gaze 在 scene camera 视频画面中的 x/y 坐标。

### 3.4 数据处理注意点

处理 `.txt` 数据时需要注意：

- 有些行可能没有 gaze 坐标。
- 有些 gaze 值可能是 `0` 或空值。
- 有些坐标可能超出视频边界。
- `.txt` 行数、有效 gaze 点数量和视频帧数可能不完全一致。
- 需要确认使用 frame index 还是 `rec_time` 对应视频时间。

导师提到视频和文本数据已经 frame-by-frame aligned，因此第一版可以优先按帧对应处理；如果发现不完全一致，再考虑基于时间戳匹配。

## 4. 输入与输出

### 4.1 输入

工具的核心输入包括：

1. `.txt` gaze data file
2. scene camera video file
3. 可选配置参数

例如：

```text
input_gaze_txt: P2_..._CsvData.txt
input_video: P2_..._Scene Cam.mp4
start_frame: 0
end_frame: 1000
brush_size: 40
opacity: 0.5
normalize: true
frame_skip: 1
```

这些配置项在早期原型中可以先通过 command-line 参数实现，但最终工具最好做成交互式控件，尽量对齐 existing Unity tool 的操作方式。

推荐 UI 控件形式：

| 配置项 | 推荐控件 | 作用 |
| --- | --- | --- |
| input gaze txt | file picker | 选择 `.txt` gaze data |
| input video | file picker | 选择 scene camera video |
| start frame / end frame | range slider 或两个 number input | 选择处理的视频范围 |
| brush size | slider | 控制 gaze 点扩散范围 |
| opacity | slider | 控制 heatmap overlay 透明度 |
| normalize heatmap | checkbox | 是否归一化 heatmap intensity |
| frame skip | slider 或 stepper | 控制处理速度和采样密度 |
| heatmap gradient | dropdown | 选择颜色映射 |
| output FPS / playback speed | slider 或 number input | 控制输出视频播放速度 |
| show gaze points | checkbox | 是否显示原始 gaze point |
| generate heatmap | button | 开始生成结果 |
| clear/reset | button | 清除当前结果 |
| export results | button | 导出图片、视频和 summary |

### 4.2 输出

工具需要输出：

1. final heatmap image：包含所选范围内全部 gaze 数据的一张总热力图。
2. heatmap development video：展示 heatmap 随时间逐步生成或变化的视频。

工具可以额外输出：

1. gaze point overlay video，用于调试和 correctness evaluation。
2. selected frame screenshots，用于报告截图。
3. summary file，用于记录输入、参数和处理速度。

summary file 可以记录：

- participant ID；
- input video path；
- video FPS；
- video resolution；
- processed frame range；
- total gaze samples；
- valid gaze samples；
- invalid gaze samples；
- brush size；
- opacity；
- frame skipping；
- processing time。

## 5. 功能模块设计

### 5.1 Data Loader

负责读取输入文件。

功能：

- 读取 `.txt` gaze data。
- 自动识别 tab-separated 格式。
- 提取 gaze x/y 列。
- 读取视频基本信息，例如 FPS、width、height、frame count。

输出：

```text
gaze_samples = [
  {frame_index, time, x, y, valid},
  ...
]
```

### 5.2 Data Cleaning

负责清理 gaze 数据。

功能：

- 删除空值。
- 删除无效值。
- 删除超出视频范围的坐标。
- 可选：记录 invalid gaze samples 的数量。

### 5.3 Gaze Point Overlay

负责将 gaze 点画到视频帧上。

用途：

- 验证 gaze 坐标是否正确。
- 作为 correctness evaluation 的证据。

第一阶段应该先完成这个模块，再做 heatmap。

### 5.4 Heatmap Generator

负责将 gaze 点转换为 heatmap。

基本思路：

1. 创建与视频帧相同大小的 heatmap matrix。
2. 对每个 gaze point，在对应位置增加一个圆形或高斯分布权重。
3. 对 heatmap 做平滑和 normalization。
4. 转换为彩色 heatmap。

重要参数：

- brush size；
- normalize；
- color map；
- frame range；
- frame skip。

### 5.5 Video Renderer

负责将 heatmap 叠加到视频帧上。

功能：

- 读取原视频帧。
- 生成或更新 heatmap。
- 将 heatmap 与原视频按 opacity 混合。
- 写出新视频。

### 5.6 Export Module

负责导出结果。

输出类型：

- `.mp4` video；
- `.png` selected frames；
- `.png` final heatmap；
- `.csv` or `.json` summary。

### 5.7 User Interface

第一版可以先用 command-line interface，因为这样最容易验证数据读取和 heatmap 算法。

例如：

```bash
python main.py --gaze P2_CsvData.txt --video P2_SceneCam.mp4 --out-dir outputs/P2 --brush-size 40 --opacity 0.5
```

但是最终工具的功能和交互方式应尽量 align with the existing Unity tool shown in the video。因此建议采用两阶段策略：

1. Week 1-4：先实现可运行的 CLI pipeline，确保数据读取、坐标验证、静态 heatmap 和动态视频输出都正确。
2. Week 5-6：在 pipeline 稳定后，增加简单 GUI 或 Streamlit 页面，将配置参数做成 sliders、checkboxes、dropdowns 和 buttons。

这样既保证项目能按时完成，又能满足“features of tool should align with existing Unity tool”的要求。

## 6. 推荐技术方案

### 6.1 编程语言

优先使用 Python。

原因：

- 导师说明 Python is preferred。
- Python 适合快速处理视频、表格数据和图像。
- 后续可以较容易加入 GUI 或 evaluation scripts。

### 6.2 推荐库

可以考虑：

- `pandas`：读取和处理 `.txt` 表格数据。
- `opencv-python`：读取视频、写视频、画点、图像混合。
- `numpy`：构建 heatmap matrix。
- `matplotlib` 或 OpenCV colormap：生成 heatmap 颜色。
- `argparse`：命令行参数。
- `time` / `json` / `csv`：记录 summary 和 performance。

### 6.3 Heatmap 生成方法

基础方法：

1. 对每个 gaze point，在 heatmap matrix 上对应坐标加 1。
2. 使用 Gaussian blur 让点扩散成热力区域。
3. 将 heatmap normalize 到 0-255。
4. 使用 color map 转成彩色图。
5. 将彩色 heatmap 按透明度叠加到原视频。

动态 heatmap 可以有两种方式：

- cumulative heatmap：从开始到当前帧不断累积。
- sliding window heatmap：只使用当前帧附近一段时间内的 gaze 点。

第一版建议先做 cumulative heatmap，因为更简单、更稳定。

## 7. 功能优先级

### 7.1 Must

必须完成：

- 读取 `.txt` gaze data。
- 读取 scene camera video。
- 提取 gaze x/y 坐标。
- 过滤无效 gaze 数据。
- 生成 gaze point overlay。
- 生成包含全部数据的 final heatmap image。
- 生成展示 heatmap development over time 的视频。
- 写 README 或使用说明。
- 做基本 correctness 和 speed evaluation。

### 7.2 Should

建议完成：

- start frame / end frame。
- brush size。
- opacity。
- normalization。
- frame skipping。
- 处理 P2-P5 多个 participant。
- 输出 summary file。
- 导出 selected frame screenshots。

### 7.3 Could

时间允许再做：

- 简单 GUI 或 Streamlit 界面，用 sliders/checkboxes/dropdowns 控制参数。
- object detector integration。
- small user evaluation。

### 7.4 Not Now

当前不做：

- JSON 数据解析。
- AR heatmap。
- 3D heatmap。
- detector training。
- detector comparison。
- full-scale user study。

## 8. 开发步骤

### Step 1: 数据检查

目标：理解 `.txt` 和视频是否能对应。

任务：

- 解压或读取 P2 数据。
- 查看 `.txt` 表头。
- 找到 gaze x/y 列。
- 检查视频 FPS、resolution、frame count。
- 检查 gaze 坐标范围是否符合视频分辨率。

产出：

- 数据检查记录。
- 初步字段说明。

### Step 2: Gaze Point Overlay Prototype

目标：验证 gaze 坐标是否正确。

任务：

- 读取前几百帧视频。
- 从 `.txt` 中读取对应 gaze 坐标。
- 在视频帧上画 gaze point。
- 输出短视频或截图。
- 与 `Original_Eye Tracking Video.mp4` 对比。

产出：

- gaze point overlay demo。
- 坐标正确性说明。

### Step 3: Basic Heatmap Image

目标：生成包含全部选定 gaze 数据的静态 heatmap。

任务：

- 对一段视频的 gaze 点进行累计。
- 生成 heatmap matrix。
- 应用 Gaussian blur。
- 应用 color map。
- 导出 final heatmap image。

产出：

- final heatmap PNG。

### Step 4: Heatmap Overlay Video

目标：生成展示 heatmap development over time 的视频。

任务：

- 对每一帧生成或更新累计 heatmap。
- 将 heatmap 与原视频叠加。
- 写出 `.mp4`。

产出：

- heatmap overlay video。

### Step 5: Configuration Options

目标：让工具可配置，并为后续 GUI 控件做准备。

任务：

- 加入 start/end frame。
- 加入 brush size。
- 加入 opacity。
- 加入 normalization。
- 加入 frame skip。
- 加入 output FPS 或 playback speed。
- 设计参数到 UI 控件的映射，例如 slider、checkbox、dropdown。

产出：

- 可配置命令行工具。
- UI 控件设计说明。

### Step 6: Multi-participant Support

目标：支持 P2-P5 数据。

任务：

- 测试 P2、P3、P4、P5。
- 检查不同 participant 的字段是否一致。
- 记录每个 participant 的有效 gaze 点数量。

产出：

- 多 participant 测试结果。

### Step 7: Documentation

目标：让别人能使用工具。

任务：

- 写 README。
- 写输入文件格式说明。
- 写参数说明。
- 写 example command。
- 写 known limitations。

产出：

- README。
- demo outputs。

### Step 8: Evaluation

目标：证明工具是正确、可用、有价值的。

任务：

- correctness evaluation；
- speed evaluation；
- optional usability/usefulness feedback。

产出：

- evaluation results；
- dissertation 中可用的表格和截图。

## 9. 8 周项目计划

### Week 1: Scope confirmation and data inspection

目标：

- 明确项目范围：只处理 Face-to-Face `.txt`，JSON 先忽略。
- 理解 Demo Data 的文件结构。
- 确认 gaze x/y 列。

任务：

- 整理项目说明。
- 检查 P2 `.txt` 文件表头。
- 检查 P2 两个视频的区别。
- 记录视频分辨率、FPS、帧数。
- 准备 GitHub repository 或项目文件结构。

交付物：

- project scope note；
- data structure note；
- initial repository。

### Week 2: Gaze point overlay prototype

目标：

- 读取 `.txt` gaze 坐标并画到视频上。

任务：

- 实现 `.txt` loader。
- 实现 video reader。
- 实现 gaze point overlay。
- 输出短视频或截图。
- 与 original eye tracking video 对比。

交付物：

- gaze point overlay prototype；
- coordinate validation screenshots。

### Week 3: Basic heatmap generation

目标：

- 从 gaze 点生成基础 heatmap。

任务：

- 创建 heatmap matrix。
- 实现 gaze accumulation。
- 实现 Gaussian blur / brush size。
- 实现 color map。
- 导出 final heatmap image。

交付物：

- final heatmap image；
- basic heatmap generation script。

### Week 4: Heatmap overlay video

目标：

- 生成展示 heatmap development over time 的视频。

任务：

- 实现 frame-by-frame heatmap overlay。
- 支持 opacity。
- 支持 cumulative heatmap。
- 导出 `.mp4`。
- 测试短视频和较长片段。

交付物：

- heatmap overlay video；
- performance notes。

### Week 5: Configuration and usability improvement

目标：

- 让工具更像一个可用 application，并逐步对齐 Unity tool 的功能和交互。

任务：

- 加入 start/end frame。
- 加入 frame skip。
- 加入 brush size 参数。
- 加入 opacity 参数。
- 加入 normalization 参数。
- 输出 summary file。
- 整理 command-line interface。
- 设计或实现简单 UI 控件，例如 sliders、checkboxes、dropdowns。

交付物：

- configurable CLI tool；
- UI prototype or UI design；
- summary output；
- updated README draft。

### Week 6: Multi-participant testing and robustness

目标：

- 让工具能处理 P2-P5。

任务：

- 测试 P2、P3、P4、P5。
- 处理空值、0 值、越界值。
- 检查不同 participant 的数据格式差异。
- 修复错误和边界情况。

交付物：

- P2-P5 output samples；
- robustness notes；
- bug fixes。

### Week 7: Evaluation

目标：

- 完成项目评估材料。

任务：

- correctness evaluation：比较 gaze overlay 与原始 red dot。
- speed evaluation：记录不同设置下处理时间。
- usability/usefulness feedback：可选地找几个人看输出或试用工具。
- 整理 evaluation 表格和截图。

交付物：

- evaluation results；
- screenshots for dissertation；
- performance table。

### Week 8: Final polishing and dissertation material

目标：

- 整理最终版本和报告材料。

任务：

- 清理代码结构。
- 完成 README。
- 准备 demo output。
- 整理 limitations 和 future work。
- 写 dissertation 中的 implementation/evaluation 部分初稿。

交付物：

- final tool；
- final README；
- demo outputs；
- dissertation draft sections。

## 10. Evaluation 设计

### 10.1 Correctness

目的：证明 gaze 坐标映射正确。

方法：

- 从 `.txt` 读取 gaze x/y。
- 生成 gaze point overlay。
- 与 original eye tracking video 中的 red dot 对比。
- 选取几个代表性帧作为证据。

可写进报告的问题：

> Does the generated gaze point appear at the same location as the reference gaze marker in the original eye-tracking video?

### 10.2 Processing Speed

目的：证明工具处理速度可接受。

可以记录：

- participant；
- processed frames；
- video duration；
- brush size；
- frame skip；
- processing time；
- frames per second processed。

### 10.3 Usability

目的：判断工具是否容易使用。

如果时间允许，可以让少量用户试用或观看结果，然后收集反馈。

可问问题：

- Is the output heatmap easy to understand?
- Are the configuration options clear?
- Is the tool useful for inspecting gaze behaviour?

### 10.4 Usefulness

目的：判断 heatmap 是否比 raw gaze point 更适合分析。

可以比较：

- 只看 red dot；
- 看 final heatmap；
- 看 heatmap overlay video。

## 11. 风险与应对

### 风险 1: 坐标列选错

应对：

- 先做 gaze point overlay。
- 与 original eye tracking video 对比。
- 不急着做 heatmap。

### 风险 2: `.txt` 行数和视频帧数不一致

应对：

- 先检查 frame count 和数据行数。
- 如果差异小，可以按 frame index 近似。
- 如果差异大，使用 `rec_time` 映射到视频时间。

### 风险 3: 无效 gaze 点很多

应对：

- 过滤空值、0 值、越界值。
- 在 summary 中记录 invalid sample 数量。

### 风险 4: 视频处理速度慢

应对：

- 加入 frame skipping。
- 只处理 start/end frame。
- 降低输出分辨率作为可选项。

### 风险 5: 项目范围变大

应对：

- 牢记当前范围：TXT video heatmap。
- JSON 和 detector 都放到 future work 或 extension。

## 12. 可以向导师确认的问题

下一次会议可以问：

1. For the Face-to-Face TXT files, should I use `Scene Cam_Original_Gaze_Gaze X/Y` as the main gaze coordinates?
2. Should I generate the heatmap on the clean `Scene Cam.mp4` and use `Original_Eye Tracking Video.mp4` only for validation?
3. Is a command-line Python tool acceptable for the first prototype?
4. Should the main output be a final heatmap image, a heatmap overlay video, or both?
5. Is object detector integration optional after the basic video heatmap works?
6. For evaluation, would a small technical evaluation plus brief usability feedback be sufficient?

## 13. 当前结论

当前最安全、最符合导师说明的路线是：

> 先忽略 JSON，只做 Face-to-Face TXT 数据的视频 heatmap。第一步验证 gaze coordinates 是否能正确画到视频上；第二步生成基础 heatmap；第三步加入配置和视频导出；第四步测试多个 participant；最后做 correctness、speed、usability/usefulness evaluation。

这个路线既符合导师“Video heatmap only works for the txt file”的说明，也能在 8 周内形成一个完整、可演示、可写进 dissertation 的毕业项目。
