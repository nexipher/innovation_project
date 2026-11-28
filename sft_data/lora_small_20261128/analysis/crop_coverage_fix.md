# TruFor 三图裁剪对编辑 mask 的覆盖率（修复前后）

| sample | base before | base after | LoRA before | LoRA after |
|---|---:|---:|---:|---:|
| `val__autosplice__40489__2__q95` | 0.265285 | 0.377611 | 0.265285 | 0.377611 |
| `val__autosplice__41993__0__q95` | 0.368519 | 0.965475 | 0.368519 | 0.965475 |
| `val__autosplice__42958__1__q95` | 0.594014 | 1.0 | 0.594014 | 1.0 |
| `val__autosplice__43686__1__q95` | 0.439543 | 1.0 | None | 1.0 |
| `val__autosplice__56086__0__q95` | 0.679454 | 0.768378 | None | 0.768378 |
| `val__autosplice__57143__0__q95` | 0.520048 | 0.981419 | 0.520048 | 0.981419 |
| `val__autosplice__57276__0__q95` | 0.305453 | 0.305453 | 0.305453 | 0.305453 |
| `val__autosplice__60788__0__q95` | 0.198558 | 0.501 | 0.198558 | 0.501 |
| `val__autosplice__67065__0__q95` | None | 1.0 | None | 1.0 |
| `val__autosplice__73275__1__q95` | None | 0.966645 | 0.262165 | 0.966645 |

覆盖率 = 交付三图内容区域覆盖编辑 mask 的比例；before 是 anomaly-peak 裁剪，after 是 high-response 包围盒裁剪。未改 TruFor 权重或阈值。
