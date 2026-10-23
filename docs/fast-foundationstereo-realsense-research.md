# Fast-FoundationStereo 与 RealSense 深度质量：官方资料核查

核查日期：2026-10-22。本文核对 NVIDIA 官方的 [Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo) 仓库，以及其 README 指向的 [FoundationStereo](https://github.com/NVlabs/FoundationStereo) 仓库。代码引用均固定到核查时的 commit，避免随默认分支变化。

- Fast-FoundationStereo：`476f4249561f7c79ca707326954f9255643412a6`
- FoundationStereo：`6e8806816b533e4d13ddbb95ffa907b797060a62`

## 结论

**仓库提供的是从左右立体图像重新估计视差/深度的方法，不是对 RealSense 已生成的 Z16 深度图做后处理。** Fast-FoundationStereo 的 README 明确说方法在立体 RGB 图像上效果最佳，也测试过 D4xx 的单色或 IR 立体图像；其 demo 输入是 `left_file` 和 `right_file`，模型推理输出视差，再用左相机焦距和双目基线换算米制深度。[Fast README：输入建议与相机参数](https://github.com/NVlabs/Fast-FoundationStereo/blob/476f4249561f7c79ca707326954f9255643412a6/readme.md#L83-L90)；[demo：双图输入和视差转深度](https://github.com/NVlabs/Fast-FoundationStereo/blob/476f4249561f7c79ca707326954f9255643412a6/scripts/run_demo.py#L64-L96) [#L114-L127](https://github.com/NVlabs/Fast-FoundationStereo/blob/476f4249561f7c79ca707326954f9255643412a6/scripts/run_demo.py#L114-L127)

因此，如果目标是比较 Fast-FoundationStereo 对现有 RealSense 深度的改善，必须从相机取得**同一时刻、配对的左右 IR 图像**，另外重建深度，再与相机原始深度对比；只把当前保存的 RGB 对齐深度图交给模型，并不符合模型输入定义。仓库的公开资料没有声称对任意场景都能比相机深度准，也没有提供针对用户当前数据的实测结果。

**官方对于精度优先场景的明确建议，是离线计算时使用原版 FoundationStereo。** Fast README 的 NOTE 说明 Fast 版本针对实时应用；若离线计算且希望获得最佳精度，应查看前作 FoundationStereo。原版 README 将 `23-51-11`（ViT-large）列为通用场景表现最好的模型，`11-33-40`（ViT-small）精度稍低但更快。[Fast README NOTE](https://github.com/NVlabs/Fast-FoundationStereo/blob/476f4249561f7c79ca707326954f9255643412a6/readme.md#L10-L13)；[FoundationStereo 权重说明](https://github.com/NVlabs/FoundationStereo/blob/6e8806816b533e4d13ddbb95ffa907b797060a62/readme.md#L61-L69)。这是一种模型和速度/精度取舍建议，仍不构成对当前 RealSense 工况必然改善的保证。

### 与本项目测试数据的对应关系

`test_depth_261022/` 当前每帧保存的是 `color.png`、对齐到 RGB 的 `d2rgb.npy`/可视化图和 metadata；目录中没有同步保存的 RealSense `infrared 1`、`infrared 2` 图像，也没有 IR 双目 baseline/完整 IR 标定记录。metadata 记录的是 D435IF 的 RGB 与 `depth_aligned_to_color` 流，且采集时 emitter 为开启、自动曝光开启。由此，这批既有文件不能直接复现仓库的左右 IR 推理，也不能仅凭 RGB 对齐深度反推出所需的 IR stereo 参数。要评估该方法，需要在相机端重新采集并保存 IR1/IR2、左右 IR 内参/畸变、IR 间外参（baseline）及帧时间戳；这些是对当前项目数据的文件核查结果，不是 Fast 仓库的额外要求。

## 仓库本身具体提供什么

| 项目 | 官方资料说明 | 对 RealSense 的含义 |
|---|---|---|
| 输入 | 左、右两个经过校正（rectified）且去畸变的图像；左图应确实来自左相机，左右不能互换。README 说立体 RGB 表现最好，也测试过 D4xx 单色/IR 立体图像。[Fast README](https://github.com/NVlabs/Fast-FoundationStereo/blob/476f4249561f7c79ca707326954f9255643412a6/readme.md#L83-L88) | 输入应为 IR1/IR2 或一对 RGB，而不是 RGB 单图或已有深度。若单通道图以二维数组传入，Fast demo 将灰度复制到三个通道。[Fast demo](https://github.com/NVlabs/Fast-FoundationStereo/blob/476f4249561f7c79ca707326954f9255643412a6/scripts/run_demo.py#L64-L84) |
| 输出 | 推理产生视差图；demo 再计算 `depth = fx * baseline / disparity`。[Fast demo](https://github.com/NVlabs/Fast-FoundationStereo/blob/476f4249561f7c79ca707326954f9255643412a6/scripts/run_demo.py#L87-L96) [#L114-L127](https://github.com/NVlabs/Fast-FoundationStereo/blob/476f4249561f7c79ca707326954f9255643412a6/scripts/run_demo.py#L114-L127) | 深度值依赖正确的左 IR 相机焦距 `fx` 和 IR 双目基线，单位应一致（demo 要求基线米）。左右 IR 的相机坐标/内参不能被 RGB 内参替代。 |
| 校正 | 左右图应去畸变并极线对齐：对应点应落在相同图像行。README 提醒高分辨率时使用层级推理或降低分辨率。[FoundationStereo README](https://github.com/NVlabs/FoundationStereo/blob/6e8806816b533e4d13ddbb95ffa907b797060a62/readme.md#L81-L89) | 应先验证 IR1 与 IR2 的图像行对齐、左右顺序、畸变处理以及使用的图像分辨率对应的内参。不能只凭“同一台相机”就假定任意分辨率和 stream 配置下的输入都已经正确校正。 |
| 后处理 | demo 可选择忽略左图中没有右图对应观测的区域，也可在生成点云后用 Open3D 半径离群点移除；它没有将该步骤描述为深度图精度增强。[Fast demo](https://github.com/NVlabs/Fast-FoundationStereo/blob/476f4249561f7c79ca707326954f9255643412a6/scripts/run_demo.py#L108-L135) | `denoise_cloud` 只改点云，不会修补或滤波原始深度图；用于点云清理时也可能删除稀疏但有效的表面点。不能把它等同于图像级深度去噪。 |

Fast model card 将输入写作 RGB 立体图像，但 Fast README 明确补充已在单色/IR 立体图像测试；demo 对单通道输入也会复制成三通道。这里以 README 的 D4xx IR 说明和实际 demo 输入路径为依据，而不是把 model card 的概述解释成“只支持 RGB”。[model card](https://github.com/NVlabs/Fast-FoundationStereo/blob/476f4249561f7c79ca707326954f9255643412a6/model_card.md#L31-L50)

## RealSense IR、投射器和曝光：资料能支持到什么程度

FoundationStereo README 的 FAQ 将“如何与 RealSense 一起运行”指向维护者在 issue 中的说明。[FAQ](https://github.com/NVlabs/FoundationStereo/blob/6e8806816b533e4d13ddbb95ffa907b797060a62/readme.md#L161-L178) 在维护者回复中，左右 IR 立体图是输入；输出深度位于左 IR 相机坐标系。若要用 RGB 图给点云着色，需根据 IR 左相机到 RGB 相机的外参重投影；用于恢复几何的相机内参应是 IR 左相机的内参，而非 RGB 内参。[维护者 issue #26 评论](https://github.com/NVlabs/FoundationStereo/issues/26#issuecomment-2726194835) [关于 IR 内参的维护者回复](https://github.com/NVlabs/FoundationStereo/issues/26#issuecomment-2727119182)。FoundationStereo issue #80 的问答还给出将灰度 IR 转为三通道输入的做法，但这些问答评论除维护者明确署名的回复外，应视为 issue 讨论而非正式校准规范。[issue #80](https://github.com/NVlabs/FoundationStereo/issues/80)

**没有找到 NVIDIA 维护者建议关闭 RealSense IR 投射器、固定或调整曝光/增益，或重新标定 RealSense 才能提高 Fast-FoundationStereo 深度精度的官方指导。** Fast 仓库的 README/demo 没有投射器或曝光选项，也没有自己的 RealSense 采集脚本。FoundationStereo 提醒的核心是图像校正、左右顺序和正确的双目参数。

Fast 仓库 issue [#57](https://github.com/NVlabs/Fast-FoundationStereo/issues/57) 中，一位非仓库维护者的评论附上了 D405 采集脚本。脚本默认关闭 emitter，并称“被动 IR 更干净”；又允许在场景缺少纹理时打开点投射器。[issue 评论及附件](https://github.com/NVlabs/Fast-FoundationStereo/issues/57#issuecomment-4726657239)。这是 issue 参与者提供的示例代码，不是 NVIDIA 维护者的明确建议，也不是仓库内的官方实现或有对照数据支撑的精度结论。因此不能据此断言 emitter off 一定更好；关闭投射器可能减少投射点干扰，也可能令低纹理区域缺少左右匹配特征，实际影响取决于场景，应在目标工况下成对采集验证。该附件脚本等待自动曝光运行 30 帧，但这同样不能推导成维护者推荐的曝光方案。

## 适用性、限制与验证重点

1. **它是另一个 stereo depth estimator，不是 Z16 修复器。** Fast-FoundationStereo 读取左右图，输出模型估计的视差；输出的噪声、无效匹配和遮挡边界错误不会因为模型“foundation”或零样本泛化而自动消失。当前 repo demo 的 `remove_invisible` 是根据 `u_right = x - disparity` 去掉左图越界的位置，不等同于完整的左右一致性/遮挡置信度验证。[源码](https://github.com/NVlabs/Fast-FoundationStereo/blob/476f4249561f7c79ca707326954f9255643412a6/scripts/run_demo.py#L108-L112)
2. **需要真实的成对 IR 帧和正确参数。** RealSense RGB、IR1、IR2 是不同视点/传感器流。左右 IR 图与时间应配对；对深度转米，需要 IR 左相机的 `fx` 及 IR baseline。若缩放输入图像，内参中的焦距、主点也必须按同样比例缩放；Fast demo 通过缩放 `K[:2]` 处理这一点。[源码](https://github.com/NVlabs/Fast-FoundationStereo/blob/476f4249561f7c79ca707326954f9255643412a6/scripts/run_demo.py#L70-L75) [#L114-L120](https://github.com/NVlabs/Fast-FoundationStereo/blob/476f4249561f7c79ca707326954f9255643412a6/scripts/run_demo.py#L114-L120)
3. **遮挡和物体边缘要单独检查。** 双目在一侧可见而另一侧不可见的表面没有有效匹配。Fast demo 能去掉一类图像边界外的对应点，但没有在示例流程中给出面向插头/槽口边缘的专门置信度过滤。应将边缘附近的视差、深度突变和点云叠加可视化，并与原相机深度及可用的参考测量逐区域比较。
4. **计算开销取决于分辨率、checkpoint 和迭代数。** Fast README 的 runtime 表是在 RTX 3090、640×480 测得；不同 checkpoint/迭代数在 PyTorch 下约 29–49 ms，在 TensorRT 下约 14–23 ms。README 还称宽度小于 1000 时效果更好，建议高分辨率时缩小输入；减少 `valid_iters` 可提速但可能降低质量，增大 `max_disp` 会更慢且更耗显存。[Fast README：速度/权衡](https://github.com/NVlabs/Fast-FoundationStereo/blob/476f4249561f7c79ca707326954f9255643412a6/readme.md#L44-L60) [#L78-L91](https://github.com/NVlabs/Fast-FoundationStereo/blob/476f4249561f7c79ca707326954f9255643412a6/readme.md#L78-L91)。该数字不是用户现有设备上的性能承诺。
5. **度量深度误差会放大视差与标定误差。** 使用 `Z = fx * B / d` 时，焦距、baseline、左右校正和视差均影响深度；越远处视差越小，相同像素视差误差会造成更大的深度误差。应先用已知距离平面/靶标校验米制深度，不能只比较彩色深度可视化的“平滑程度”。

## 对当前项目可做的后续小规模试验

若后续要判断是否值得用于当前插头/槽口任务，可以先离线做一个小试验，不改主流程：在同一 RealSense 帧组里保存同步的原始 IR1/IR2、相机输出 Z16、对应配置/内参/外参与 RGB；确认配对、分辨率、左右顺序和极线校正后，分别计算 stereo disparity/depth 与相机深度。对静态场景分别测 emitter on/off（如硬件允许），保持曝光策略可记录并尽量可比；用已知距离或几何量具做局部误差统计，并重点检查插头边缘、黑色/弱纹理、反光区域和遮挡边界。只有对目标距离与 ROI 的误差、有效覆盖率和帧间稳定性都优于现有深度时，才可称为质量提升。

这是建议的验证设计，不是仓库已经验证过的结果，也不意味着 Fast-FoundationStereo 能直接修复项目当前保存的 aligned Z16 图。

## 一手来源索引

- [NVlabs/Fast-FoundationStereo README（固定 commit）](https://github.com/NVlabs/Fast-FoundationStereo/blob/476f4249561f7c79ca707326954f9255643412a6/readme.md)
- [Fast-FoundationStereo demo（固定 commit）](https://github.com/NVlabs/Fast-FoundationStereo/blob/476f4249561f7c79ca707326954f9255643412a6/scripts/run_demo.py)
- [Fast-FoundationStereo model card（固定 commit）](https://github.com/NVlabs/Fast-FoundationStereo/blob/476f4249561f7c79ca707326954f9255643412a6/model_card.md)
- [NVlabs/FoundationStereo README（固定 commit）](https://github.com/NVlabs/FoundationStereo/blob/6e8806816b533e4d13ddbb95ffa907b797060a62/readme.md)
- [FoundationStereo RealSense 相机参数与坐标系 issue #26](https://github.com/NVlabs/FoundationStereo/issues/26)
- [FoundationStereo IR 图像输入 issue #80](https://github.com/NVlabs/FoundationStereo/issues/80)
- [Fast-FoundationStereo D4xx 问题 issue #57](https://github.com/NVlabs/Fast-FoundationStereo/issues/57)
