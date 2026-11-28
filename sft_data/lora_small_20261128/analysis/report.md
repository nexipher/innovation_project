# 15 张 AutoSplice 验证图 — TruFor 独立分析

判读：score、异常图/mask 的 IoU/recall、三图裁剪对 mask 的覆盖率，以及
base/LoRA 的实际 local 标签。分类：not_called / weak_or_mislocalized /
panels_miss_region / usable_but_agent_missed / detected；真图为
true_negative / false_positive。

| sample | group | score | IoU | recall | base crop | LoRA crop | base class | lora class |
|---|---|---:|---:|---:|---:|---:|---|---|
| `val__autosplice__40489__2__q95` | AutoSplice_Forged | 0.443609 | 0.127245 | 0.127245 | 0.265285 | 0.265285 | weak_or_mislocalized | weak_or_mislocalized |
| `val__autosplice__41993__0__q95` | AutoSplice_Forged | 0.816312 | 0.74799 | 0.764828 | 0.368519 | 0.368519 | panels_miss_region | panels_miss_region |
| `val__autosplice__42958__1__q95` | AutoSplice_Forged | 0.999071 | 0.892361 | 0.913047 | 0.594014 | 0.594014 | detected | detected |
| `val__autosplice__43686__1__q95` | AutoSplice_Forged | 0.561782 | 0.773682 | 0.774963 | 0.439543 | None | panels_miss_region | not_called |
| `val__autosplice__56086__0__q95` | AutoSplice_Forged | 0.256417 | 0.362366 | 0.363958 | 0.679454 | None | weak_or_mislocalized | not_called |
| `val__autosplice__57143__0__q95` | AutoSplice_Forged | 0.932441 | 0.865482 | 0.87879 | 0.520048 | 0.520048 | detected | usable_but_agent_missed |
| `val__autosplice__57276__0__q95` | AutoSplice_Forged | 0.16367 | 0.0 | 0.0 | 0.305453 | 0.305453 | weak_or_mislocalized | weak_or_mislocalized |
| `val__autosplice__60788__0__q95` | AutoSplice_Forged | 0.412761 | 0.340128 | 0.341872 | 0.198558 | 0.198558 | weak_or_mislocalized | weak_or_mislocalized |
| `val__autosplice__67065__0__q95` | AutoSplice_Forged | 0.991293 | 0.912046 | 0.931993 | None | None | not_called | not_called |
| `val__autosplice__73275__1__q95` | AutoSplice_Forged | 0.785686 | 0.605941 | 0.608545 | None | 0.262165 | not_called | panels_miss_region |
| `val__autosplice__42307__authentic__q95` | AutoSplice_Authentic | 0.364614 | None | None | None | None | true_negative | true_negative |
| `val__autosplice__47630__authentic__q95` | AutoSplice_Authentic | 0.151429 | None | None | None | None | true_negative | true_negative |
| `val__autosplice__52191__authentic__q95` | AutoSplice_Authentic | 0.069942 | None | None | None | None | true_negative | not_called |
| `val__autosplice__65749__authentic__q95` | AutoSplice_Authentic | 0.117181 | None | None | None | None | true_negative | true_negative |
| `val__autosplice__70102__authentic__q95` | AutoSplice_Authentic | 0.18563 | None | None | None | None | true_negative | true_negative |
