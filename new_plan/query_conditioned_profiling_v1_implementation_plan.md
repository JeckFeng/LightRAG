# Query-Conditioned Profiling V1 代码改造实施计划

## 摘要

本计划严格遵循 [`new_plan/query_conditioned_profiling_v1_design.md`](/mnt/data_nvme/code/LightRAG/new_plan/query_conditioned_profiling_v1_design.md) 的阶段划分，只分为 `Phase 1: 离线 profile 入库`、`Phase 2: 接入 local 查询链`、`Phase 3: 验证与 ablation`。
执行目标是实现实体侧、`local` 模式、固定 facet schema 约束下的 Query-Conditioned Profiling V1，且保持现有 `entity -> graph node + entities_vdb` 主链不被推翻，`description` 继续作为 fallback。

## 关键接口与约束

- 新增存储属性：
  - `LightRAG.entity_profiles`
  - `LightRAG.entity_profiles_vdb`
- 新增 `QueryParam` 字段：
  - `enable_entity_profiles`
  - `entity_profile_top_k`
  - `entity_profile_max_per_entity`
- 新增函数：
  - `_parse_entity_profile_generation_result`
  - `_generate_entity_profiles`
  - `_upsert_entity_profiles`
  - `_select_profiles_for_entities`
  - `_compose_entity_profile_description`
- 需要扩展参数透传的函数：
  - `_merge_nodes_then_upsert`
  - `merge_nodes_and_edges`
  - `_get_node_data`
  - `_perform_kg_search`
  - `kg_query`
  - `_build_query_context`
- 严格边界：
  - 只做实体侧，不做关系侧
  - 只接 `local`，不接 `global / hybrid / mix`
  - 不删除 graph node 上现有 `description`
  - 不把 profile 塞进 graph storage
  - 只预留 `support_chunk_ids / support_fragment_ids / grounding_status`，不实现 fragment-level grounding

## 分阶段实施

### Phase 1：只把数据存起来

#### 任务包 1：定义 namespace、常量和 schema

- 在 `namespace.py` 增加：
  - `KV_STORE_ENTITY_PROFILES`
  - `VECTOR_STORE_ENTITY_PROFILE`
- 在 `constants.py` 增加：
  - `DEFAULT_ENTITY_PROFILE_SCHEMA_ID`
  - `DEFAULT_ENTITY_PROFILE_FACETS`
  - `DEFAULT_ENTITY_PROFILE_SCHEMA_VERSION`
  - `DEFAULT_ENTITY_PROFILE_TOP_K`
  - `DEFAULT_ENTITY_PROFILE_MAX_PER_ENTITY`
- 在 `base.py` 紧跟 `TextChunkSchema` 后新增：
  - `EntityFacetSchemaItem`
  - `EntityProfileSchema`
  - `EntityProfilesRecordSchema`
- 扩展 `QueryParam`，但此阶段只定义字段，不接查询逻辑。

验收标准：

- 新增类型和常量可被其他模块导入。
- schema 字段名与设计文档完全一致。

#### 任务包 2：接 storage wiring 和配置校验

- 在 `lightrag.py` 初始化：
  - `self.entity_profiles: BaseKVStorage`
  - `self.entity_profiles_vdb: BaseVectorStorage`
- 在 `initialize_storages()`、`finalize_storages()`、`_insert_done()` 中纳入新存储。
- 加入配置校验：
  - `enable_entity_profiles=True` 时 `entity_profile_facets` 不能为空
  - 每个 facet 必须有 `facet_id / facet_name / definition`
  - `facet_id` 唯一
  - `entity_profile_default_facet_id` 必须存在于 facet schema 中

验收标准：

- 开启开关时系统能正常初始化新存储。
- 错误配置会在启动阶段失败，而不是运行中失败。

#### 任务包 3：实现离线 profile 生成 prompt 与解析

- 在 `prompt.py` 新增：
  - `entity_profile_generation_system_prompt`
  - `entity_profile_generation_user_prompt`
- 实现 `_parse_entity_profile_generation_result`：
  - 只接受 `profile` 类型记录
  - 以本地 `facet_catalog` 校验 `facet_id`
  - 回填标准化 profile 结构
- 解析结果必须附带：
  - `profile_id`
  - `facet_definition`
  - `support_chunk_ids`
  - `support_fragment_ids`
  - `grounding_status`
  - `created_at`

验收标准：

- 模型输出无法引入未声明 facet。
- parser 输出结构可直接写入 KV/VDB。

#### 任务包 4：实现 profile 生成与落库主链

- 在 `operate.py` 实现：
  - `_generate_entity_profiles`
  - `_upsert_entity_profiles`
- `_generate_entity_profiles` 负责：
  - 基于 `description_list + base_description + facet_catalog` 生成 facet-specific profiles
  - 使用 `llm_response_cache`
  - 对缺失 facet 做 fallback 补全
- `_upsert_entity_profiles` 负责：
  - 组装 `EntityProfilesRecordSchema`
  - 写 `entity_profiles`
  - 写 `entity_profiles_vdb`
  - 删除 stale profile vector ids
- 修改 `_merge_nodes_then_upsert`：
  - 在 graph node 和 `entities_vdb` upsert 完成后，按开关调用 `_upsert_entity_profiles`
- 修改 `merge_nodes_and_edges`：
  - 透传新 storage 参数

验收标准：

- 插入文档后，每个实体除了原有 graph node / entities_vdb 外，还会生成 `entity_profiles` 与 `entity_profiles_vdb` 数据。
- fallback 行为稳定，不会因 LLM 漏 facet 导致空 profile 集合。

#### 任务包 5：补齐 PostgreSQL 后端兼容

- 在 `lightrag/kg/postgres_impl.py` 的 `NAMESPACE_TABLE_MAP` 增加：
  - `LIGHTRAG_ENTITY_PROFILES`
  - `LIGHTRAG_VDB_ENTITY_PROFILE`
- 将新 namespace 接入 KV 和 vector 相关分支逻辑。

验收标准：

- PostgreSQL 后端不会因新增 namespace 报未知表或未知 namespace 错误。
- 默认 JSON/NanoVectorDB 路径和 PostgreSQL 路径都能持有同一组 profile 数据语义。

### Phase 2：接到 local 查询链

#### 任务包 6：实现在线 profile 选择与组合

- 在 `operate.py` 实现：
  - `_select_profiles_for_entities`
  - `_compose_entity_profile_description`
- `_select_profiles_for_entities` 负责：
  - 在已召回 `node_datas` 内部做第二阶段选择
  - 使用 `entity_profiles_vdb.query()`
  - 仅保留属于已召回实体的 profile
  - 对每个实体最多保留 `entity_profile_max_per_entity`
  - 保持同一实体内部的 facet 多样性
- `_compose_entity_profile_description` 负责：
  - 按本地 facet schema 顺序组合 `selected_profiles`
  - 无选中 profile 时回退 `base_description`

验收标准：

- 查询链不会直接替换实体召回，只会在召回结果内做二次 profile 选择。
- `description` 组合结果稳定、可读、可回退。

#### 任务包 7：把 profile 选择接入 local 查询路径

- 修改 `_get_node_data`：
  - 增加 `entity_profiles_storage`、`entity_profiles_vdb`、`apply_profiles`
  - 在组装 `node_datas` 后、关系查找前执行 profile 选择
- 修改 `_perform_kg_search`：
  - 仅在 `local` 分支传 `apply_profiles=True`
  - `hybrid / mix` 保持 `apply_profiles=False`
- 修改 `kg_query` 和 `_build_query_context`：
  - 继续透传新 storage 参数
- 不改 `_build_context_str()` 的主体结构，只让上游 `description` 被条件化替换。

验收标准：

- `local` 模式下 entity 的 `description` 变为 facet-schema-constrained 的条件化描述。
- 非 `local` 模式行为与原版保持一致。

#### 任务包 8：暴露调试与 API 可观察性

- 在 `utils.py` 的 `convert_to_user_format()` 增加可选返回字段：
  - `base_description`
  - `selected_profile_ids`
  - `selected_facet_ids`
  - `selected_profiles`

验收标准：

- API / raw_data 可直接观察 profile 选择结果。
- 不破坏现有字段结构和老调用方兼容性。

### Phase 3：验证与 ablation

#### 任务包 9：补齐单测

- 新增 `tests/test_entity_profile_generation.py`
  - 覆盖 profile record 生成
  - 覆盖 KV/VDB 数量一致性
  - 覆盖 `facet_id / support_chunk_ids / grounding_status`
- 新增 `tests/test_entity_profile_local_query.py`
  - 覆盖 `enable_entity_profiles=False` 回退原行为
  - 覆盖 `enable_entity_profiles=True` 时存在 `selected_profiles`
  - 覆盖 `local` 模式 description 被条件化改写
  - 覆盖 profile 缺失 fallback 到 `base_description`
  - 覆盖 facet 只落在预定义 schema 内

验收标准：

- 新测试能覆盖离线入库链和 local 查询链两个闭环。
- 开关关闭时行为与原版一致。

#### 任务包 10：执行 ablation 验证

本任务包不再采用通用数据集验证，而是收敛为一份专门面向 HotpotQA 的最小闭环实验计划。

最小闭环约束如下：

- 只使用：
  - `/mnt/data_nvme/code/LightRAG/datasets/hotpotqa/hotpot_dev_distractor_v1.json`
- 只做：
  - `bridge` 样本
- 样本规模按三阶段推进：
  - `100` 条 bridge
  - `300` 条 bridge
  - 全部 bridge
- 只评估：
  - 官方 `Exact Match`
  - 官方 `F1`
- 只做实体侧方法验证，不扩展到关系侧 profile。
- 只比较 profile 形式，不改变同题的原始 context、实体抽取流程、回答模型和评分方式。

##### 10.1 HotpotQA 最小闭环验证目标

- 本轮验证目标不是复现开放域 HotpotQA 全流程，而是在严格控制变量的前提下验证：
  - “实体侧 query-conditioned profile 是否比原始静态 profile 更有用”
- 该验证采用 question-specific closed-world 设定：
  - 每条样本只对自己的 `context` 建图
  - 不跨样本共享语料
  - 不混入全局 Wikipedia
- 该设定的价值：
  - 最大限度降低开发成本
  - 排除开放域召回噪声
  - 先把方法闭环跑通，再决定是否扩展到更完整设定

##### 10.2 根目录 Ablation 文件组织架构

- 从本阶段开始，所有消融实验相关代码、配置、清单、结果都统一放在仓库根目录：
  - `/mnt/data_nvme/code/LightRAG/Ablation`
- 原始 HotpotQA 文件不复制、不改写，始终直接读取：
  - `/mnt/data_nvme/code/LightRAG/datasets/hotpotqa/hotpot_dev_distractor_v1.json`
- `Ablation` 建议目录结构如下：

```text
Ablation/
  README.md
  hotpotqa_minimal_closed_loop/
    README.md
    configs/
      subset_bridge_100.json
      subset_bridge_300.json
      subset_bridge_all.json
      experiment_static_single_profile.json
      experiment_multi_profile_fixed.json
      experiment_multi_profile_query_conditioned.json
    manifests/
      bridge_100_case_manifest.jsonl
      bridge_300_case_manifest.jsonl
      bridge_all_case_manifest.jsonl
    preprocessing/
      select_hotpot_bridge_subset.py
      build_case_corpus.py
    pipelines/
      run_case_indexing.py
      run_case_query.py
      run_hotpot_ablation.py
    modes/
      static_single_profile.py
      multi_profile_fixed.py
      multi_profile_query_conditioned.py
    evaluation/
      export_hotpot_predictions.py
      eval_hotpot_em_f1.py
    results/
      bridge_100/
        static_single_profile/
        multi_profile_fixed/
        multi_profile_query_conditioned/
      bridge_300/
        static_single_profile/
        multi_profile_fixed/
        multi_profile_query_conditioned/
      bridge_all/
        static_single_profile/
        multi_profile_fixed/
        multi_profile_query_conditioned/
    logs/
```

- 目录职责约束：
  - `configs/`：保存子集规模配置与实验配置
  - `manifests/`：保存筛选后的样本清单，不复制原始 HotpotQA JSON
  - `preprocessing/`：保存数据读取、bridge 过滤、样本级 document 构建逻辑
  - `pipelines/`：保存整套批处理执行入口
  - `modes/`：保存三组消融模式的 query-time 适配逻辑
  - `evaluation/`：保存官方 `EM/F1` 兼容导出与评测封装
  - `results/`：保存预测、评分、汇总表
  - `logs/`：保存运行日志和失败重试记录

##### 10.3 HotpotQA 数据处理流程

- 第 1 步：读取 `hotpot_dev_distractor_v1.json`
- 第 2 步：过滤出：
  - `sample["type"] == "bridge"`
- 第 3 步：根据固定顺序或固定随机种子生成三个子集清单：
  - `bridge_100`
  - `bridge_300`
  - `bridge_all`
- 第 4 步：将每条样本转换为 case manifest，至少保留以下字段：
  - `_id`
  - `question`
  - `answer`
  - `type`
  - `level`
  - `supporting_facts`
  - `context`
- 第 5 步：把每条样本的 `context` 转换为“本题专属语料库”
  - `context` 的每个元素格式为 `[title, sentences]`
  - 每个元素必须转换为一个独立段落级 document
  - 严禁把整条样本的全部段落拼成一个大文档
- 第 6 步：段落级 document 的最小结构建议为：

```json
{
  "case_id": "5a8...",
  "doc_id": "5a8...__03",
  "title": "Justice League (film)",
  "paragraph_text": "Justice League is an upcoming American superhero film ...",
  "content": "Justice League (film)\nJustice League is an upcoming American superhero film ...",
  "sentence_count": 4,
  "is_supporting_title": true
}
```

- 第 7 步：其中 `content` 字段作为真正送入 LightRAG 的文档内容：
  - 格式固定为 `title + "\\n" + paragraph_text`
  - 这样既保留标题实体信号，又避免把多段混成单文档

##### 10.4 单题建图与隔离策略

- 每条 HotpotQA 样本都必须单独建图。
- 每个问题只对自己的 `context` 建图，禁止：
  - 跨样本共享工作目录
  - 把上一题索引残留带入下一题
  - 把全局 Wikipedia 或其他 Hotpot 文件混入当前题
- 每题单独工作目录建议为：
  - `Ablation/hotpotqa_minimal_closed_loop/results/<subset>/<mode>/workspaces/<case_id>/`
- 每题处理流程固定为：
  - 创建新 workspace
  - 将当前题所有 paragraph documents 插入 LightRAG
  - 等待索引完成
  - 在该 workspace 内执行 query
  - 导出结果
- 这样做的目的不是模拟最终系统，而是保证：
  - 变量隔离最强
  - 出错样本可单独复跑
  - 能直接定位 profile 选择是否影响答案

##### 10.5 三组消融实验设置

- 本阶段固定只做以下三组消融：

| mode_id | 名称 | 说明 |
|---|---|---|
| `static_single_profile` | 静态单 profile | 使用原始 LightRAG entity description，不启用 entity profiles |
| `multi_profile_fixed` | 多视角 profile，但不做 query-conditioned 选择 | 使用离线生成的多视角 profiles，但 query 时不做选择，按固定顺序直接拼接 |
| `multi_profile_query_conditioned` | 多视角 profile + query-conditioned selection | 使用当前实现的 query-conditioned entity profile |

- 三组实验的控制变量要求：
  - 同一条样本
  - 同一份 `context`
  - 同一套实体/关系抽取
  - 同一个回答 LLM
  - 同一 `local` 查询模式
  - 同一 `top_k/chunk_top_k/max_*_tokens`
  - 同一批 query
- 唯一变化项：
  - entity profile 的使用方式

##### 10.6 三组模式的实现约束

- `static_single_profile`
  - 直接使用：
    - `QueryParam(mode="local", enable_entity_profiles=False)`
  - 不读取 `selected_profiles`
- `multi_profile_query_conditioned`
  - 直接使用：
    - `QueryParam(mode="local", enable_entity_profiles=True)`
  - 使用当前主线实现的 profile 选择逻辑
- `multi_profile_fixed`
  - 这是本阶段新增的消融模式，当前主线代码没有现成实现
  - 必须在 `Ablation/hotpotqa_minimal_closed_loop/modes/multi_profile_fixed.py` 中实现
  - 实现原则：
    - 允许复用离线生成好的 `entity_profiles`
    - query 时不做向量召回式 profile 选择
    - 直接按 facet schema 固定顺序组合多视角 profiles
    - 该模式用于回答：
      - 提升是否仅来自“多写几段摘要”
      - 提升是否真正来自 query-conditioned 选择
- `multi_profile_fixed` 的组合建议：
  - 按 facet schema 顺序拼接所有可用 facets
  - 输出格式与 query-conditioned 组保持一致：
    - `[facet_id] profile_text`
  - 不允许按 query 动态丢 facet

##### 10.7 100 -> 300 -> 全部 bridge 的执行策略

- 第一阶段：
  - `bridge_100`
  - 目标：只验证流程能跑通，确认 JSON 处理、单题建图、三组消融、预测导出、EM/F1 评分全链路无阻塞
- 第二阶段：
  - `bridge_300`
  - 目标：在更稳定的样本量下观察三组模式排序是否一致
- 第三阶段：
  - `bridge_all`
  - 目标：输出最终 HotpotQA 最小闭环验证结果
- 执行顺序必须严格为：
  - `bridge_100 -> bridge_300 -> bridge_all`
- 禁止在 `bridge_100` 未跑通前直接跳到 `bridge_300` 或 `bridge_all`

##### 10.8 单样本运行流程

- 对每个 `case_id`，执行以下固定流程：
  - 读取样本
  - 解析并构造 paragraph-level documents
  - 新建 case workspace
  - 插入当前题 documents
  - 运行 `static_single_profile`
  - 运行 `multi_profile_fixed`
  - 运行 `multi_profile_query_conditioned`
  - 保存三组答案和中间 raw_data
- 每组输出至少包含：
  - `case_id`
  - `question`
  - `gold_answer`
  - `pred_answer`
  - `raw_data`
  - `mode_id`
  - `workspace_path`
- 对 profile 组额外保存：
  - `base_description`
  - `selected_profile_ids`
  - `selected_facet_ids`
  - `selected_profiles`

##### 10.9 官方 EM/F1 输出格式

- 为兼容 HotpotQA 官方评测脚本，必须导出官方兼容预测文件。
- 每个子集、每个实验模式都输出一个 predictions 文件，建议命名为：
  - `predictions_<subset>_<mode>.json`
- 建议最小格式如下：

```json
{
  "answer": {
    "5a8b57f25542995d1e6f1371": "yes",
    "5adce1755542990d50227d50": "Justice League"
  },
  "sp": {}
}
```

- 说明：
  - 当前最小闭环只验证答案质量，因此 `answer` 是核心输出
  - 若官方脚本要求完整字段结构，则保留空的 `sp`
  - 最终汇总时只记录 answer-level `EM/F1`
- 每个子集、每个模式都应再输出一个评分摘要文件，建议命名为：
  - `metrics_<subset>_<mode>.json`
- 摘要至少包含：

```json
{
  "subset": "bridge_100",
  "mode": "multi_profile_query_conditioned",
  "count": 100,
  "em": 0.0000,
  "f1": 0.0000
}
```

##### 10.10 结果记录模板

- 数据集清单表：

| subset_id | source_file | filter_rule | sample_count | manifest_path | notes |
|---|---|---|---|---|---|
| bridge_100 | `hotpot_dev_distractor_v1.json` | `type == "bridge"` | 100 | 待填写 | 待填写 |

- 单样本结果表：

| case_id | subset | mode | question | gold_answer | pred_answer | em | f1 | selected_facet_ids | notes |
|---|---|---|---|---|---|---|---|---|---|
| 5adce1755542990d50227d50 | bridge_100 | `static_single_profile` | 待填写 | Justice League | 待填写 | 0/1 | 0-1 | `[]` | 待填写 |

- 三组对比表：

| case_id | static_em | fixed_em | conditioned_em | static_f1 | fixed_f1 | conditioned_f1 | best_mode | notes |
|---|---|---|---|---|---|---|---|---|
| 5adce1755542990d50227d50 | 0/1 | 0/1 | 0/1 | 0-1 | 0-1 | 0-1 | 待填写 | 待填写 |

- 子集汇总表：

| subset | mode | sample_count | em | f1 | avg_selected_profile_count | fallback_ratio | notes |
|---|---|---|---|---|---|---|---|
| bridge_100 | `static_single_profile` | 100 | 待填写 | 待填写 | 0 | N/A | 待填写 |

##### 10.11 Ablation 执行清单

- 第 1 步：在 `Ablation/` 下创建 HotpotQA 最小闭环实验目录结构。
- 第 2 步：实现 `select_hotpot_bridge_subset.py`，只从 `hotpot_dev_distractor_v1.json` 读取并筛选 bridge。
- 第 3 步：生成 `bridge_100 / bridge_300 / bridge_all` 三份 manifest。
- 第 4 步：实现 `build_case_corpus.py`，把每条样本的 `context` 转成 paragraph-level documents。
- 第 5 步：实现单题单 workspace 建图流程。
- 第 6 步：实现三组模式运行器：
  - `static_single_profile`
  - `multi_profile_fixed`
  - `multi_profile_query_conditioned`
- 第 7 步：先跑 `bridge_100`，确认三组模式都能产出预测结果。
- 第 8 步：导出 `predictions_bridge_100_<mode>.json`
- 第 9 步：执行官方兼容 `EM/F1` 评测并保存摘要。
- 第 10 步：确认 `bridge_100` 跑通后，再扩到 `bridge_300`。
- 第 11 步：确认 `bridge_300` 指标和流程稳定后，再扩到 `bridge_all`。
- 第 12 步：完成最终汇总表和结论。

##### 10.12 必须回答的实验问题

- `multi_profile_query_conditioned` 是否优于 `static_single_profile`
- `multi_profile_fixed` 是否优于 `static_single_profile`
- `multi_profile_query_conditioned` 是否优于 `multi_profile_fixed`
- 如果 `multi_profile_fixed` 和 `multi_profile_query_conditioned` 差异明显，是否可以说明提升主要来自 query-conditioned 选择，而不只是“多写几段摘要”
- 如果 `multi_profile_fixed` 已有明显提升，是否可以说明“多视角离线 profile”本身就有价值
- 在 HotpotQA bridge 设定下，当前方法是否值得继续扩展到下一阶段副创新

##### 10.13 本阶段交付物

- `Ablation/hotpotqa_minimal_closed_loop/` 目录及其脚本骨架
- 三份 bridge 子集 manifest
- 三组模式的 predictions 文件
- 三组模式的 `EM/F1` 摘要文件
- 子集级汇总表
- 一份文字化实验结论，明确：
  - 哪组最好
  - 提升来自哪里
  - 是否值得继续扩大实验

验收标准：

- 满足设计文档第 11 节的 6 条“V1 跑通”判定标准。
- 可以支撑后续 `+ Evidence-Grounded Composition` 的消融基线。

## 测试与检查清单

- 单元测试：
  - profile parser
  - profile 生成 fallback
  - KV/VDB 写入一致性
  - local query profile 选择
- 集成检查：
  - 文档插入后 `entity_profiles` 与 `entity_profiles_vdb` 均有数据
  - `local` 下能看到 `selected_profiles`
  - `global / hybrid / mix` 不受影响
- 回归检查：
  - 关闭 `enable_entity_profiles` 时行为退回原版
  - `description` 仍保留为稳定 fallback
  - PostgreSQL namespace 兼容不报错

## 默认假设

- 计划文档输出粒度采用“任务包级”，每个阶段拆成可直接分配和验收的任务。
- 设计文档中的默认 4-facet schema、字段名、fallback 规则、函数命名和阶段顺序全部视为冻结要求，不在实施计划中重新设计。
- 计划产物建议保存为：
  - `/mnt/data_nvme/code/LightRAG/new_plan/query_conditioned_profiling_v1_implementation_plan.md`
- 当前计划不包含实际写文件动作；若切回执行模式，按上述路径落盘即可。
