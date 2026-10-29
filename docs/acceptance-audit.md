# 申报书对照与验收自查（2026-10-30）

本表同时对照工作区的 `moonhnsw-申报书.md` 和
[2026 MoonBit 黑客松十月赛程公布的验收标准](https://moonbitlang.github.io/Hackathon2026/)。
官方页面列出六项验收标准；最终执行口径仍以正式章程和官方通知为准。
数字是可复现的项目证据，不构成任意环境下的性能承诺。

## 最终验收指南九项（参赛者提供）

| 要求 | 可核查证据 | 自查 |
| --- | --- | --- |
| 1. MoonBit 为主要语言，`moonc >= 0.10.16` | 核心库、功能测试及基准计算均为 MoonBit；本机 `moonc -v` 为 `v0.10.17`，CI 有最低版本校验。 | 满足；Node 脚本仅下载数据并启动 JS 基准。 |
| 2. GitHub 公开，提交清晰 | [公开仓库](https://github.com/yiwnagmolu/moonhnsw)、`git log --oneline --decorate`、版本标签和 `CHANGELOG.md`。 | 满足；功能、测试、基准、文档及发布记录可追溯。 |
| 3. 结构清晰，核心功能完整 | `hnsw.mbt`、`insert.mbt`、`search.mbt`、`distance.mbt`、`serial.mbt` 等按职责拆分；下方逐项映射申报功能。 | 核心 HNSW 构建、检索与持久化可运行；近似搜索及单线程边界在 README 明示。 |
| 4. README 目标、安装、用法和示例可复现 | `README.mbt.md` 的 Quick start 含 `moon add`、包级 `moon.pkg` 导入、完整 `main` 和运行命令；全新消费者项目从 Mooncakes 安装验证。 | 满足。 |
| 5. CI 覆盖检查、构建、测试 | `.github/workflows/ci.yml` 执行 `moon fmt --check`、`moon check --target wasm-gc --deny-warn`、三个后端的测试、wasm 构建、文档和示例。 | 满足；以对应提交的 GitHub Actions 结果为准。 |
| 6. 可运行示例 | `examples/benchmark` 由 CI 执行；README 另有最小消费者示例。 | 满足。 |
| 7. 核心路径测试 | 83 项黑盒/白盒测试，覆盖三种距离、构建检索、删除压实、量化、持久化、过滤和异常输入；wasm-gc、JS 及 CI native 执行。 | 已覆盖声明的核心路径；测试无法证明零缺陷。 |
| 8. Mooncakes 发布 | `moon view yiwnagmolu/moonhnsw --json` 可查询公开包和最新版本；README 有安装命令。 | 满足。 |
| 9. OSI 许可证及参考来源合规 | `LICENSE` 和 `moon.mod` 均为 [OSI 认可的 MIT](https://opensource.org/license/mit)；README 列出 HNSW 论文和参数约定的参考项目，说明为独立实现；UCI 数据在基准报告中署名和注明 CC BY 4.0。 | 仓库未包含参照库代码或原始数据；历史创作过程仍须由参赛者说明。 |

以上是代码和公开记录层面的验收自查；参赛者对技术选择与 AI 辅助工作的
解释，以及最终是否通过验收，由组委会评定。

## 赛程验收标准

| 官方标准 | 仓库证据与自查 |
| --- | --- |
| MoonBit 为主 | HNSW 索引、距离、量化、持久化、搜索及基准计算均在 `.mbt` 文件中；Node 启动器负责下载数据和运行 JS 构建。 |
| 仓库公开，提交可追踪 | [公开仓库](https://github.com/yiwnagmolu/moonhnsw)保留功能、测试、文档、基准和发布提交及版本标签；本次新增工作在本页下表列明。 |
| README、示例、必要测试可运行 | README 给出全新 MoonBit 项目的安装与包导入步骤；`examples/` 有演示；本机 wasm-gc 和 JS 测试各 83 项，CI 覆盖 native。 |
| 本期有实质新增 | v0.12.0 增加精确搜索、SQ8 等能力和真实数据基准；v0.12.2 再补第三组数据、三种度量交叉测量、边界回归及可复现记录。详见 `CHANGELOG.md` 与 Git 提交。 |
| 开源许可证与来源说明 | MIT 许可证、模块元数据与版权行一致；库为原创实现，未复制参考实现；UCI 数据来源、许可及使用方式在基准报告中注明。 |
| AI 辅助成果可解释 | README、源码 API、测试、基准方法及局限均可供参赛者逐项复核和解释。对参赛者现场解释能力的最终判断只能由组委会作出。 |

官方同时说明重复提交、拆分项目或仅做简单修改不计入验收。以上仅是
公开材料和本仓库的自查，不代表组委会的正式验收结论。

## 申报计划完成度

| 申报事项 | 当前证据 | 结论与边界 |
| --- | --- | --- |
| 纯 MoonBit HNSW，三种距离及可调参数 | `hnsw.mbt`、`insert.mbt`、`search.mbt`、`distance.mbt`；`api_test.mbt` 与 `hnsw_wbtest.mbt` | 核心索引无 FFI；JS 基准的数据入口使用 FFI，不属于库核心。 |
| 增量插入、top-k、半径检索、删除与压实 | `insert.mbt`、`search.mbt`、`lifecycle.mbt`；对应白盒和黑盒测试 | 半径图搜索仍是近似；需要完整性时使用 `search_radius_exact`。压实会重建索引并改变内部 id。 |
| 标签、自省、持久化及旧格式兼容 | `hnsw.mbt`、`lifecycle.mbt`、`serial.mbt`；标签、统计、CRC、v1–v4 载荷测试 | 当前写入格式 v5；旧载荷只能由较新版本读取。持久化为全量保存。 |
| 扩展能力 | `adaptive.mbt`、`calibrate.mbt`、`exact.mbt`、`quant.mbt`；黑盒测试 | 自适应过滤和校准只对给定索引及查询有经验意义；SQ8 的“精确”指对量化存储值精确。 |
| 扩展功能与工程量 | `search.mbt` 的 MMR、批量和过滤检索；`lifecycle.mbt` 的合并、深拷贝、图导出与收缩；13 个库源码文件约 2,700 行、7 个测试文件约 2,211 行（含空行及注释） | 行数由当前跟踪的 `.mbt` 文件统计，只是规模描述，质量以行为验证和审查为准。 |
| 真实基准数据 | [`real-benchmarks.md`](real-benchmarks.md) 与三份原始日志；UCI 原始文件 URL、SHA-256、划分、环境、方法均列出 | Pen Digits、Letter Recognition、Wine Quality 三组均在 v0.12.2 以 L2、InnerProduct、Cosine 运行；均为 JS release、单台 Windows 机器、200 个留出查询。Wine Quality 的 SQ8 tie-aware recall@10 为 0.9285，不能宣传固定的 0.99 召回率。 |
| 大规模与三种度量 | `examples/benchmark` 覆盖三种度量；`examples/benchmark-large` 覆盖 100k L2；README 标注运行版本 | 100k 结果来自旧版手动运行；尚无当前版本、三个后端、三种度量的完整交叉矩阵。不得将旧运行当作当前版普遍性能。 |
| 边界与回归 | `moon test --target wasm-gc` 和 `moon test --target js` 各 83/83；CI 另跑 native、wasm 构建、文档和格式检查 | 涵盖空索引、全删除、重复向量、非有限输入、等距排序、标签更新、SQ8 压实、有效 CRC 的恶意载荷等。当前工作树的 native 测试由 GitHub CI 验证，本机缺 MSVC。 |
| 安装与使用说明 | README 快速开始；在全新项目中 `moon add`、包级 `moon.pkg` 导入、编译与运行测试均已验证 | `moon add` 只声明模块依赖；使用 `@moonhnsw` 的包仍需显式导入。 |

## 提交、许可证与来源

- 至 v0.12.3 有 80 个提交，发布历史按功能、基准、文档与版本提交分开；可用 `git log --oneline --decorate` 和 `git tag --list` 核对。v0.12.0 的功能与真实基准提交分别是 `8551872`、`e479027`；v0.12.2 的验收自查和基准提交是 `28a47c3`。
- `LICENSE` 是 MIT 许可证，`moon.mod` 的 `license` 为 `MIT`。Git 提交作者为 `yiwnagmolu`；版权行标明维护者和贡献者。MIT 文本要求分发副本保留版权与许可声明。
- 核心包只导入 `moonbitlang/core` 的 `math`、`debug`、`double`；基准启动器只使用 Node.js 内建模块。仓库不包含第三方实现代码或 UCI 原始 CSV。申报书声明 HNSW 代码为独立实现；仓库审查能确认无 vendored 参照实现，但无法单凭仓库证明所有历史创作过程。
- 三份 UCI 数据集页面均标为 CC BY 4.0；基准报告给出作者、DOI、源链接和使用方式。运行器按固定 SHA-256 校验下载内容，数据缓存在仓库之外。
- 发布前用 `git ls-files` 检查跟踪文件，未发现凭据、私钥或原始数据文件。此检查只覆盖当前跟踪文件，不代替专门的历史密钥扫描。

## 复核命令

```sh
moon info
moon fmt --check
moonc -v
moon check --target wasm-gc --deny-warn
moon test --target wasm-gc
moon test --target js
moon build --target wasm
moon doc
node --check benchmarks/real/run.cjs
node benchmarks/real/run.cjs pendigits
node benchmarks/real/run.cjs letters
node benchmarks/real/run.cjs winequality
git log --oneline --decorate -20
git tag --list
```

本机缺少 MSVC，native 测试由 `.github/workflows/ci.yml` 在 Linux CI 运行。
真实基准下载 UCI 文件并检查 SHA-256；若源文件变化，运行器会中止，
需要人工核验并更新校验和后才能重新报告数据。
