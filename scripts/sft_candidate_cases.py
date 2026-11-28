"""Candidate assistant turns for the first-round SFT exemplars.

Each entry is content, not a template of what the model should conclude.  The
runner turns it into protocol-shaped turns and executes the named experts for
real.  Numbers quoted in the text are the screening measurements of the same
deterministic experts; `{{TRUFOR_ID}}` is filled from the token actually
delivered.  No ground-truth label or mask is used in any assistant turn.
"""

CASES = {
    # ------------------------------------------------------------------
    # AutoSplice forged: local tampering on a photographic scene
    # ------------------------------------------------------------------
    "autosplice__54029__1__q95": {
        "call_order": ["trufor", "probe"],
        "first": {
            "region": "none",
            "visual": "户外树荫下的晾衣绳，挂着多件衣物；右侧一件白色衣物表面很亮，下缘与背景的分界较直，左侧树冠、草地和远处水面保持自然连续。",
            "anomalies": "右侧白色衣物的亮度和边界与周围柔软布料不太一致，值得先做局部测量；其余区域没有明显异常。",
            "target": "trufor — 先确认右侧白色衣物附近是否存在局部内容变化；这张图整体像照片，整图来源问题稍后再补测。",
        },
        "second": {
            "region": "[300, 760, 935, 910]",
            "visual": "同一张晾衣绳照片；右侧白色衣物是画面中最亮的对象，衣物褶皱和背景树影仍可见。",
            "anomalies": "TruFor 分数 0.9995，异常图的高响应集中在右侧白色衣物的一条竖直区域，可靠性图在同一区域为高可靠；响应没有落在树干或草地上。整图来源仍未测量。",
            "target": "probe — 局部高响应已经出现，但整图是否 AI 生成仍无测量；PROBE 是唯一的整图见证，补测后才能把整图来源和局部篡改分开写。",
        },
        "reasoning": "TruFor 0.9995，异常图的高响应落在右侧白色衣物，可靠性图在同一区域为高可靠，原图也显示该衣物下缘与背景的分界较直；这支持局部内容变化。PROBE 0.7219 高于 0.5，但它是整图判别器的原始分数，不定位、也不是校准概率；原图其余场景的照片连续性只是视觉观察，不能替代独立的整图来源测量。整图来源因此保留 Uncertain，不写成 Real；局部 Tampered 支撑总体 Fake。",
        "verdict": {
            "verdict": "Fake", "whole_image_generation": "Uncertain",
            "local_manipulation": "Tampered",
            "suspicious_regions": [{
                "bbox": [300, 760, 935, 910],
                "reason": "TruFor 高响应集中在右侧白色衣物，可靠性图同区域可靠，原图该处内容边界较直",
                "evidence": ["{{TRUFOR_ID}}"],
            }],
            "certainty": "High",
            "primary_evidence": ["trufor_expert"],
            "report": "TruFor 高响应局限在右侧白色衣物且可靠，支持局部篡改；PROBE 只有整图高分、没有独立的整图来源支撑，整图 Uncertain，总体由局部篡改判 Fake。",
        },
    },
    "autosplice__55310__1__q95": {
        "call_order": ["trufor", "probe"],
        "first": {
            "region": "[180, 300, 475, 505]",
            "visual": "旧信箱和瓷砖墙，信箱口夹着一张印有“MAGLINE”字样的报纸；报纸矩形轮廓清楚，信箱木板有磨损和阴影。",
            "anomalies": "报纸的矩形边界与信箱口、墙面之间显得过于整齐，像一块独立内容；先测量该局部区域。",
            "target": "trufor — 报纸区域是最明显的局部候选，先确认异常响应和可靠性。",
        },
        "second": {
            "region": "[180, 300, 475, 505]",
            "visual": "同一张信箱照片；报纸仍在信箱口，表面文字和图像清晰。",
            "anomalies": "TruFor 分数 0.9992，异常图的高响应正好覆盖报纸矩形，可靠性图在该矩形内为高可靠；信箱木板和瓷砖没有同等响应。整图来源尚未测量。",
            "target": "probe — 局部证据已足够具体，但整图来源仍空白；补测 PROBE 以判断这是整图生成还是照片中的局部内容变化。",
        },
        "reasoning": "TruFor 0.9992，异常图高响应与报纸矩形重合，可靠性图在同一区域高可靠；原图显示报纸作为独立矩形内容块出现在信箱口，边界与木板、瓷砖不连续，支持局部篡改。PROBE 0.0782 低于阈值，说明整图判别器倾向 Real；低 PROBE 不能清除局部高响应，二者回答不同问题。整图判 Real，局部判 Tampered，总体 Fake。",
        "verdict": {
            "verdict": "Fake", "whole_image_generation": "Real",
            "local_manipulation": "Tampered",
            "suspicious_regions": [{
                "bbox": [180, 300, 475, 505],
                "reason": "TruFor 高响应覆盖报纸矩形，可靠性高，原图该矩形与信箱表面不连续",
                "evidence": ["{{TRUFOR_ID}}"],
            }],
            "certainty": "High",
            "primary_evidence": ["trufor_expert"],
            "report": "TruFor 的高响应与报纸矩形重合且可靠，支持局部篡改；PROBE 低分只说明整图倾向 Real，不能清除局部证据。",
        },
    },
    "autosplice__45749__1__q95": {
        "call_order": ["probe", "trufor"],
        "first": {
            "region": "none",
            "visual": "冰面上的企鹅，背景是蓝色冰块和雪地，企鹅身体边缘、羽毛和冰面纹理都可见；画面整体像一张自然照片。",
            "anomalies": "企鹅身体与冰面的交界处略显得整齐，但没有足以单独判定的局部证据；先测量整图来源，再决定是否补测局部。",
            "target": "probe — 画面整体像照片，但需要整图判别器回答“整图是否 AI 生成”；局部问题暂时保留。",
        },
        "second": {
            "region": "[315, 330, 730, 650]",
            "visual": "同一张企鹅照片；企鹅位于画面中央偏左，冰面和雪地纹理连续。",
            "anomalies": "PROBE 0.9994 很高，倾向 AI-generated；但它是整图分数、不定位。为判断高分是否来自企鹅这一局部对象，补测 TruFor。",
            "target": "trufor — PROBE 的高分不能说明响应来自哪里；用局部定位检查企鹅身体是否有集中响应，以区分整图生成和局部内容变化。",
        },
        "reasoning": "PROBE 0.9994 是整图判别器的原始分数，高于阈值但不定位；它既不能证明整张图生成，也不能被局部证据自动解释成整图来源。TruFor 0.9983，异常图的高响应集中在企鹅身体，可靠性图同区域高可靠，冰面和雪地没有同等响应；原图冰面纹理、雪地阴影保持连续，但这只是视觉观察，不能替代独立的整图来源测量。局部证据支持企鹅这一内容块的变化，局部判 Tampered；整图来源仍缺独立支撑，判 Uncertain；总体由局部篡改支撑为 Fake。",
        "verdict": {
            "verdict": "Fake", "whole_image_generation": "Uncertain",
            "local_manipulation": "Tampered",
            "suspicious_regions": [{
                "bbox": [315, 330, 730, 650],
                "reason": "TruFor 高响应集中在企鹅身体，可靠性高，冰面和雪地无同等响应",
                "evidence": ["{{TRUFOR_ID}}"],
            }],
            "certainty": "High",
            "primary_evidence": ["trufor_expert"],
            "report": "TruFor 高响应局限在企鹅身体且可靠，支持局部篡改；PROBE 高分没有独立整图来源支撑，整图 Uncertain，总体由局部篡改判 Fake。",
        },
    },
    "autosplice__69868__0__q95": {
        "call_order": ["trufor", "probe"],
        "first": {
            "region": "[100, 360, 1000, 720]",
            "visual": "草地上的老虎，身体横跨画面中部，皮毛条纹、头部和四肢可见；背景是虚化的黄绿色植被。",
            "anomalies": "老虎身体轮廓与背景之间的过渡偏硬，皮毛细节在身体中部显得过于平整；先测量该区域。",
            "target": "trufor — 老虎身体是主要局部候选，先看异常图和可靠性图是否支持内容变化。",
        },
        "second": {
            "region": "[100, 360, 1000, 720]",
            "visual": "同一张老虎照片；老虎占据画面中部，背景植被虚化连续。",
            "anomalies": "TruFor 0.9952，异常图的高响应覆盖老虎身体大部，可靠性图在该区域较高；背景植被没有同等响应。整图来源仍未测量。",
            "target": "probe — 局部响应明确，但整图是否生成仍空白；补测 PROBE 以完成整图来源判断。",
        },
        "reasoning": "TruFor 0.9952，异常图高响应覆盖老虎身体，可靠性图在该区域较高，原图老虎轮廓与背景过渡偏硬，支持局部内容变化。PROBE 0.8101 高于阈值，但它是整图分数、不能定位；背景植被和地面的照片连续性只是视觉观察，不能替代独立的整图来源测量。局部判 Tampered；整图来源缺独立支撑，判 Uncertain；总体由局部篡改支撑为 Fake。局部框较宽，定位精度有限，certainty 取 Medium。",
        "verdict": {
            "verdict": "Fake", "whole_image_generation": "Uncertain",
            "local_manipulation": "Tampered",
            "suspicious_regions": [{
                "bbox": [100, 360, 1000, 720],
                "reason": "TruFor 高响应覆盖老虎身体，可靠性较高，原图轮廓与背景过渡偏硬",
                "evidence": ["{{TRUFOR_ID}}"],
            }],
            "certainty": "Medium",
            "primary_evidence": ["trufor_expert"],
            "report": "TruFor 高响应覆盖老虎身体且可靠性较高，支持局部篡改；PROBE 高分没有独立整图来源支撑，整图 Uncertain，总体由局部篡改判 Fake。",
        },
    },
    "autosplice__48251__0__q95": {
        "call_order": ["probe", "trufor"],
        "first": {
            "region": "none",
            "visual": "室内人像，一名年轻男性面对镜头，背景是暗色墙面和少量设备；人物头发、皮肤和衣领细节可见。",
            "anomalies": "人物头部与背景的交界较硬，但没有足以单独判定的局部证据；先测量整图来源。",
            "target": "probe — 人像的整图来源是首要问题；先看整图判别器，再决定是否补测局部。",
        },
        "second": {
            "region": "[30, 310, 1000, 730]",
            "visual": "同一张室内人像；人物位于画面右侧，暗色背景和左侧设备保持连续。",
            "anomalies": "PROBE 0.9308 高于阈值，倾向 AI-generated；但它不定位。为判断高分是否来自人物头部/肩部这一局部区域，补测 TruFor。",
            "target": "trufor — PROBE 的整图高分无法说明局部来源；检查人物头部和肩部是否有集中响应。",
        },
        "reasoning": "PROBE 0.9308 高于阈值，但它是整图判别器的原始分数，不定位，不能单独证明整图生成。TruFor 0.9934，异常图高响应覆盖人物头部和肩部，可靠性图在该区域较高；背景墙面和左侧设备没有同等响应，原图人物边缘与背景过渡偏硬。局部证据支持人物这一内容块的变化，局部判 Tampered；整图来源缺独立支撑，判 Uncertain；总体由局部篡改支撑为 Fake。局部框较宽，定位只到头部/肩部范围，精确边界仍有限。",
        "verdict": {
            "verdict": "Fake", "whole_image_generation": "Uncertain",
            "local_manipulation": "Tampered",
            "suspicious_regions": [{
                "bbox": [30, 310, 1000, 730],
                "reason": "TruFor 高响应覆盖人物头部和肩部，可靠性较高，原图人物边缘与背景过渡偏硬",
                "evidence": ["{{TRUFOR_ID}}"],
            }],
            "certainty": "Medium",
            "primary_evidence": ["trufor_expert"],
            "report": "TruFor 高响应集中在人物头部和肩部，支持局部篡改；PROBE 高分没有独立整图来源支撑，整图 Uncertain，总体由局部篡改判 Fake。",
        },
    },
    "autosplice__72548__1__q95": {
        "call_order": ["trufor", "probe"],
        "first": {
            "region": "none",
            "visual": "黑白舞蹈照片，舞者在浅色墙面前抬臂、屈身，身体轮廓和墙面光影可见；画面颗粒和对比度较高。",
            "anomalies": "舞者身体与墙面之间的边缘较硬，但整体像一张舞台照片；先做局部测量。",
            "target": "trufor — 舞者身体是主要局部候选，先看异常响应和可靠性。",
        },
        "second": {
            "region": "[60, 100, 950, 620]",
            "visual": "同一张黑白舞蹈照片；舞者占据画面左侧到中部，墙面和地面保持连续。",
            "anomalies": "TruFor 0.9942，异常图的高响应沿舞者身体分布，可靠性图在身体区域中等偏高；响应范围较宽，定位不够精确。整图来源尚未测量。",
            "target": "probe — 局部响应存在但定位较宽，仍需整图来源判断；补测 PROBE。",
        },
        "reasoning": "TruFor 0.9942，异常图高响应沿舞者身体分布，可靠性图在身体区域中等偏高；原图舞者轮廓与墙面之间过渡较硬。该响应支持局部内容变化，但范围宽、定位精度有限。PROBE 0.8643 高于阈值，但它是整图分数、不定位；墙面和地面的光影连续性只是视觉观察，不能替代独立的整图来源测量。局部判 Tampered；整图来源缺独立支撑，判 Uncertain；总体由局部篡改支撑为 Fake。certainty 取 Medium 以反映定位限制。",
        "verdict": {
            "verdict": "Fake", "whole_image_generation": "Uncertain",
            "local_manipulation": "Tampered",
            "suspicious_regions": [{
                "bbox": [60, 100, 950, 620],
                "reason": "TruFor 高响应沿舞者身体分布，可靠性中等偏高，原图身体与墙面过渡较硬",
                "evidence": ["{{TRUFOR_ID}}"],
            }],
            "certainty": "Medium",
            "primary_evidence": ["trufor_expert"],
            "report": "TruFor 高响应沿舞者身体分布，支持局部篡改，但范围较宽；PROBE 高分没有独立整图来源支撑，整图 Uncertain，总体由局部篡改判 Fake。",
        },
    },
    "autosplice__64223__2__q95": {
        "call_order": ["trufor", "probe"],
        "first": {
            "region": "[45, 265, 930, 755]",
            "visual": "泥土地面上的一个红色容器和蓝色物件，周围有枯枝、落叶和碎石；容器边缘和地面纹理可见。",
            "anomalies": "容器和蓝色物件像一块独立内容，边缘与地面纹理的过渡较硬；先测量该区域。",
            "target": "trufor — 容器/物件区域是主要局部候选，先确认异常响应和可靠性。",
        },
        "second": {
            "region": "[45, 265, 930, 755]",
            "visual": "同一张地面物件照片；红色容器和蓝色物件位于画面中部，周围枯枝和泥土保持连续。",
            "anomalies": "TruFor 0.9934，异常图的高响应覆盖容器和物件区域，可靠性图在该区域较高；周围泥土和枯枝没有同等响应。整图来源尚未测量。",
            "target": "probe — 局部证据明确，但整图来源仍空白；补测 PROBE 以判断是否为整图生成。",
        },
        "reasoning": "TruFor 0.9934，异常图高响应覆盖容器和蓝色物件，可靠性图在该区域较高，原图该内容块与泥土、枯枝的过渡较硬，支持局部内容变化。PROBE 0.5434 仅略高于阈值，是整图分数且不定位；周围地面纹理的照片连续性只是视觉观察，不能替代独立的整图来源测量。局部判 Tampered；整图来源缺独立支撑，判 Uncertain；总体由局部篡改支撑为 Fake。PROBE 的边界高分作为限制写入报告，不把它当作整图生成的证明。",
        "verdict": {
            "verdict": "Fake", "whole_image_generation": "Uncertain",
            "local_manipulation": "Tampered",
            "suspicious_regions": [{
                "bbox": [45, 265, 930, 755],
                "reason": "TruFor 高响应覆盖容器和物件区域，可靠性较高，原图该内容块与地面过渡较硬",
                "evidence": ["{{TRUFOR_ID}}"],
            }],
            "certainty": "Medium",
            "primary_evidence": ["trufor_expert"],
            "report": "TruFor 高响应覆盖容器和物件区域且可靠，支持局部篡改；PROBE 仅略高于阈值且无独立整图来源支撑，整图 Uncertain，总体由局部篡改判 Fake。",
        },
    },
    "autosplice__45914__2__q95": {
        "call_order": ["probe", "trufor"],
        "first": {
            "region": "none",
            "visual": "土路上的旧公交车，车身横跨画面中部，车顶、车窗和轮胎可见；背景是树木、电线杆和灰白色天空。",
            "anomalies": "公交车与背景的交界处略硬，但没有足以单独判定的局部证据；先测量整图来源。",
            "target": "probe — 场景整体像照片，但需要整图判别器回答是否 AI 生成；局部问题稍后补测。",
        },
        "second": {
            "region": "[200, 190, 735, 945]",
            "visual": "同一张公交车照片；车身占据画面中部偏右，土路、树木和电线杆保持连续。",
            "anomalies": "PROBE 0.9899 高于阈值，倾向 AI-generated，但不定位；为判断高分是否来自公交车这一内容块，补测 TruFor。",
            "target": "trufor — 检查公交车车身是否有集中高响应，以区分整图生成和局部内容变化。",
        },
        "reasoning": "PROBE 0.9899 是整图分数，高于阈值但不定位，不能证明整图生成，也不能被局部证据自动解释成整图来源。TruFor 0.9928，异常图高响应覆盖公交车车身，可靠性图在该区域较高；土路、树木和电线杆没有同等响应，原图这些背景保持照片连续性，但这只是视觉观察，不能替代独立的整图来源测量。局部判 Tampered；整图来源缺独立支撑，判 Uncertain；总体由局部篡改支撑为 Fake。",
        "verdict": {
            "verdict": "Fake", "whole_image_generation": "Uncertain",
            "local_manipulation": "Tampered",
            "suspicious_regions": [{
                "bbox": [200, 190, 735, 945],
                "reason": "TruFor 高响应覆盖公交车车身，可靠性较高，背景道路和树木无同等响应",
                "evidence": ["{{TRUFOR_ID}}"],
            }],
            "certainty": "High",
            "primary_evidence": ["trufor_expert"],
            "report": "TruFor 高响应集中在公交车车身且可靠，支持局部篡改；PROBE 高分没有独立整图来源支撑，整图 Uncertain，总体由局部篡改判 Fake。",
        },
    },

    # ------------------------------------------------------------------
    # AutoSplice authentic: high TruFor scores that must not become Tampered
    # ------------------------------------------------------------------
    "autosplice__51186__authentic__q95": {
        "call_order": ["trufor", "probe"],
        "first": {
            "region": "none",
            "visual": "海边岩石上的乌鸦，背景是水面、远山和天空；画面右侧有文字“Canuck the Crow / Vancouver, B.C. Canada”，右下角有日期水印。",
            "anomalies": "乌鸦本体和背景没有明显异常，但画面右缘有一条亮边，先做局部测量确认它是否是内容变化。",
            "target": "trufor — 先检查右缘亮边和乌鸦本体是否有局部高响应；整图来源稍后补测。",
        },
        "second": {
            "region": "none",
            "visual": "同一张乌鸦照片；乌鸦、水面和远山保持自然连续，右缘是画框边缘。",
            "anomalies": "TruFor 0.9723 很高，但异常图的高响应只出现在最右侧一条窄边，可靠性图在该窄边为高可靠；乌鸦本体、岩石和水面没有高响应。整图来源尚未测量。",
            "target": "probe — 局部高分落在画框边缘而不是内容上；补测 PROBE 以确认整图来源，并判断该高分是否只是边缘响应。",
        },
        "reasoning": "TruFor 0.9723 高于阈值，但异常图的高响应只出现在最右侧一条窄边，可靠性图在同一条边缘为高可靠；乌鸦本体、岩石、水面和文字区域都没有高响应。原图右缘就是画框边界，该响应不落在内容变化上。PROBE 0.0008 低于阈值，支持整图 Real。局部不写 Tampered：高分有，但支持它的响应在画框边缘而非内容；局部判 Not tampered，整图 Real，总体 Real。certainty 取 Medium：结论有支持，但 TruFor 高分本身仍是残余限制。",
        "verdict": {
            "verdict": "Real", "whole_image_generation": "Real",
            "local_manipulation": "Not tampered",
            "suspicious_regions": [],
            "certainty": "Medium",
            "primary_evidence": ["probe_dino_expert", "trufor_expert"],
            "report": "TruFor 高分仅来自画框右缘的窄条响应，乌鸦和背景内容无高响应；PROBE 低分支持整图 Real，局部不构成 Tampered，certainty Medium。",
        },
    },
    "autosplice__65547__authentic__q95": {
        "call_order": ["trufor", "probe"],
        "first": {
            "region": "[473, 323, 1000, 1000]",
            "visual": "室内人像，一名留脏辫的男子侧脸面向左侧，穿黄色连帽衫；背景是红色墙面，人物肩部和衣领有明显褶皱。",
            "anomalies": "人物右下肩部和衣领的褶皱较复杂，先做局部测量；画面没有明显的文字或拼接边。",
            "target": "trufor — 先检查肩部/衣领区域的异常响应和可靠性，再决定整图来源。",
        },
        "second": {
            "region": "[470, 320, 1000, 1000]",
            "visual": "同一张人像；黄色连帽衫的肩部、衣领和背景红墙保持连续。",
            "anomalies": "TruFor 0.8972 高于阈值，异常图的高响应集中在右下肩部和衣领附近，可靠性图在该区域中等偏高；但原图该处是衣物褶皱和墙面交界，没有清楚的内容替换边界。整图来源尚未测量。",
            "target": "probe — 局部响应存在但内容解释不唯一；补测 PROBE 先确认整图来源，避免把衣物褶皱直接写成篡改。",
        },
        "reasoning": "TruFor 0.8972 高于阈值，异常图的高响应集中在右下肩部和衣领，可靠性图中等偏高；但原图该区域是黄色衣物的褶皱、阴影与红墙交界，没有清楚的内容替换边界或文字异常。按协议，高分区域只是候选，不能仅凭亮度写成 Tampered；局部判 Uncertain 并说明缺的是内容层面的支持。PROBE 0.0372 低于阈值，支持整图 Real。总体因局部未解决而判 Uncertain，certainty 为 null。",
        "verdict": {
            "verdict": "Uncertain", "whole_image_generation": "Real",
            "local_manipulation": "Uncertain",
            "suspicious_regions": [{
                "bbox": [470, 320, 1000, 1000],
                "reason": "TruFor 高响应在肩部/衣领，但原图该处是衣物褶皱和墙面交界，缺少内容层面的替换证据",
                "evidence": ["{{TRUFOR_ID}}"],
            }],
            "certainty": None,
            "primary_evidence": ["probe_dino_expert", "trufor_expert"],
            "report": "PROBE 低分支持整图 Real，但 TruFor 在肩部/衣领的高响应只能算候选，原图缺少内容替换证据，局部保留 Uncertain。",
        },
    },
    "autosplice__57332__authentic__q95": {
        "call_order": ["trufor", "probe"],
        "first": {
            "region": "none",
            "visual": "灰色混凝土地面上摆放着拆解的枪械部件，金属表面有划痕和反光，背景是粗糙的墙面。",
            "anomalies": "枪械部件的金属边缘较亮，但没有明显的拼接或文字异常；先做局部测量。",
            "target": "trufor — 先检查金属部件附近是否有局部高响应，再决定是否需要整图来源判断。",
        },
        "second": {
            "region": "none",
            "visual": "同一张枪械部件照片；金属划痕、反光和混凝土地面保持连续。",
            "anomalies": "TruFor 0.6269 高于阈值，但异常图的高响应只出现在画面底边和右缘的窄条；裁剪布局显示底边响应落在画面下半部裁剪的底边，对应整图底边；可靠性图整体较高，响应不在枪械主体上。整图来源尚未测量。",
            "target": "probe — 整图来源尚未测量；PROBE 是整图来源的判别器，补测以完成整图来源判断。",
        },
        "reasoning": "核对裁剪布局：面板显示的是画面下半部区域 [500,122,1000,494]，高响应落在裁剪底边，也就是整图底边；另一个小响应在整图右缘。两者都是整图边缘，不是内容区的异常。按与 51186 相同的口径，边缘响应不作为局部篡改证据；局部判 Not tampered。PROBE 0.4577 低于阈值，倾向整图 Real，整图判 Real；总体 Real。certainty 取 Medium：TruFor 分数仍高于阈值，边缘响应是残余限制。",
        "verdict": {
            "verdict": "Real", "whole_image_generation": "Real",
            "local_manipulation": "Not tampered",
            "suspicious_regions": [],
            "certainty": "Medium",
            "primary_evidence": ["probe_dino_expert", "trufor_expert"],
            "report": "TruFor 高分只落在整图底边和右缘，不是内容区；PROBE 低于阈值支持整图 Real，局部 Not tampered，certainty Medium。",
        },
    },
    "autosplice__70684__authentic__q95": {
        "call_order": ["trufor", "probe"],
        "first": {
            "region": "[461, 541, 719, 754]",
            "visual": "雪地上的北极熊，熊身偏黄白色，脚下是积雪和水面，背景是冰块；毛发和雪地纹理可见。",
            "anomalies": "北极熊身体中部的亮度较高，先做局部测量；画面没有文字或明显拼接边。",
            "target": "trufor — 先检查北极熊身体附近是否有局部高响应和可靠性支持。",
        },
        "second": {
            "region": "[461, 541, 719, 754]",
            "visual": "同一张北极熊照片；熊身、积雪和水面保持连续。",
            "anomalies": "TruFor 0.5883 略高于阈值，异常图的高响应落在北极熊身体中部，可靠性图在该区域较高；但响应占比较小，原图熊毛与雪地过渡自然，没有清楚的内容替换边界。整图来源尚未测量。",
            "target": "probe — 局部响应略高于阈值但内容支持有限；补测 PROBE 确认整图来源，避免把轻微响应写成篡改。",
        },
        "reasoning": "TruFor 0.5883 仅略高于阈值，异常图的高响应落在北极熊身体中部，可靠性图较高，但高响应占比小；原图熊毛、积雪和水面过渡自然，没有清楚的内容替换边界。按协议，分数高于阈值但只有候选时局部写 Uncertain，并说明缺的是内容层面的支持。PROBE 0.0022 低于阈值，支持整图 Real。总体因局部未解决判 Uncertain，certainty 为 null。",
        "verdict": {
            "verdict": "Uncertain", "whole_image_generation": "Real",
            "local_manipulation": "Uncertain",
            "suspicious_regions": [{
                "bbox": [461, 541, 719, 754],
                "reason": "TruFor 高响应落在北极熊身体中部，但响应占比小，原图毛皮与雪地过渡自然",
                "evidence": ["{{TRUFOR_ID}}"],
            }],
            "certainty": None,
            "primary_evidence": ["probe_dino_expert", "trufor_expert"],
            "report": "PROBE 低分支持整图 Real；TruFor 仅略高于阈值，响应落在熊身但缺少内容层面的替换证据，局部保留 Uncertain。",
        },
    },
    "autosplice__52769__authentic__q95": {
        "call_order": ["trufor", "probe"],
        "first": {
            "region": "[52, 273, 465, 672]",
            "visual": "穿深色西装、戴眼镜的男士半身像，背景是美国国旗和蓝色旗帜；人物面部、衬衫领口和西装纹理清晰。",
            "anomalies": "人物面部亮度较高，先做局部测量；背景旗帜和西装没有明显异常。",
            "target": "trufor — 先检查面部区域的异常响应，并重点看同一区域的可靠性图。",
        },
        "second": {
            "region": "[52, 273, 465, 672]",
            "visual": "同一张男士半身像；面部、眼镜和西装领口保持连续。",
            "anomalies": "TruFor 0.5810 略高于阈值，异常图的高响应落在面部，但可靠性图在同一面部区域明显偏低（蓝色/青色），说明该处响应不可靠；背景旗帜没有高响应。整图来源尚未测量。",
            "target": "probe — 面部响应与低可靠性同时出现；补测 PROBE 确认整图来源，再决定局部是否只能保留 Uncertain。",
        },
        "reasoning": "TruFor 0.5810 仅略高于阈值，异常图的高响应落在面部；但可靠性图在同一面部区域明显偏低，按协议不能把低可靠区域的亮度当成篡改证据。原图面部、眼镜和西装领口连续，没有清楚的内容替换边界。局部判 Uncertain，并明确指出限制是“同一区域可靠性低”，不是泛泛的无法证明。PROBE 0.0937 低于阈值，支持整图 Real。总体因局部未解决判 Uncertain，certainty 为 null。",
        "verdict": {
            "verdict": "Uncertain", "whole_image_generation": "Real",
            "local_manipulation": "Uncertain",
            "suspicious_regions": [{
                "bbox": [52, 273, 465, 672],
                "reason": "TruFor 面部高响应所在区域的可靠性图偏低，响应不能作为篡改证据",
                "evidence": ["{{TRUFOR_ID}}"],
            }],
            "certainty": None,
            "primary_evidence": ["probe_dino_expert", "trufor_expert"],
            "report": "PROBE 低分支持整图 Real；TruFor 面部响应所在区域可靠性低，不能据此写 Tampered，局部保留 Uncertain。",
        },
    },
    "autosplice__67006__authentic__q95": {
        "call_order": ["probe", "trufor"],
        "first": {
            "region": "none",
            "visual": "老式黑白/泛黄的人像照片，一名戴军帽、穿制服的男子，照片边缘有磨损和相纸质感；人物面部和制服细节可见。",
            "anomalies": "照片整体像老相片，没有明显的现代拼接边；先测量整图来源。",
            "target": "probe — 先回答整图是否 AI 生成；老照片质感与整图分数可能不一致，需要记录这个冲突。",
        },
        "second": {
            "region": "[250, 360, 415, 570]",
            "visual": "同一张老式人像；男子面部、军帽和制服保持连续，照片边缘磨损可见。",
            "anomalies": "PROBE 0.5527 仅略高于阈值，倾向 AI-generated，但分数接近边界；为检查面部是否有局部响应，补测 TruFor。",
            "target": "trufor — PROBE 只是边界高分，且老照片质感与其不一致；补测局部，看面部是否有集中响应。",
        },
        "reasoning": "PROBE 0.5527 仅略高于阈值，是整图判别器的原始分数，不定位也不是校准概率；原图的相纸磨损、泛黄色调和制服细节更像老照片，但不能用外观直接推翻分数。TruFor 0.4281 低于阈值，异常图在面部有一个小范围高响应，可靠性图较高；该响应很小，原图面部、军帽和制服连续，没有清楚的内容替换边界。整图因 PROBE 边界高分而保留 Uncertain，局部因 TruFor 低于阈值但仍有小候选而保留 Uncertain；总体 Uncertain，certainty 为 null。",
        "verdict": {
            "verdict": "Uncertain", "whole_image_generation": "Uncertain",
            "local_manipulation": "Uncertain",
            "suspicious_regions": [{
                "bbox": [250, 360, 415, 570],
                "reason": "TruFor 在面部有小范围高响应但总分低于阈值，原图没有清楚的内容替换边界",
                "evidence": ["{{TRUFOR_ID}}"],
            }],
            "certainty": None,
            "primary_evidence": ["probe_dino_expert", "trufor_expert"],
            "report": "PROBE 仅边界高分且与老照片质感不一致，TruFor 低于阈值但面部有小候选；整图与局部都保留 Uncertain。",
        },
    },

    # ------------------------------------------------------------------
    # GenImage whole-generated
    # ------------------------------------------------------------------
    "genimage__SD15__459_sdv5_00195__q95": {
        "call_order": ["trufor", "probe"],
        "first": {
            "region": "[242, 166, 902, 818]",
            "visual": "棚拍风格的人像，模特穿黑色蕾丝内衣，浅色背景，皮肤和布料纹理平滑；画面构图干净，光线均匀。",
            "anomalies": "内衣的蕾丝边缘和肩带显得非常锐利，先做局部测量；整图来源稍后补测。",
            "target": "trufor — 先检查内衣/肩带区域的异常响应和可靠性，确认是否存在局部内容变化。",
        },
        "second": {
            "region": "[242, 166, 902, 818]",
            "visual": "同一张棚拍人像；模特、内衣和浅色背景保持连续。",
            "anomalies": "TruFor 0.9799，异常图的高响应覆盖内衣和肩带，可靠性图在该区域较高；但画面整体皮肤、背景和光线都非常平滑。整图来源尚未测量。",
            "target": "probe — 局部响应很强，但还需要整图来源判断；补测 PROBE，确认这是整图生成还是照片中的局部编辑。",
        },
        "reasoning": "TruFor 0.9799 的高响应覆盖内衣和肩带，可靠性图在该区域较高；但原图皮肤、背景和光线整体平滑，构图和材质更像整图生成。PROBE 0.9999 接近 1，支持整图 AI-generated。按协议，整图判 AI-generated 后局部问题为 Not applicable：内衣区域的高响应属于整图生成内容的一部分，不能再写成一次独立的局部篡改；总体由整图生成支撑为 Fake。",
        "verdict": {
            "verdict": "Fake", "whole_image_generation": "AI-generated",
            "local_manipulation": "Not applicable",
            "suspicious_regions": [],
            "certainty": "High",
            "primary_evidence": ["probe_dino_expert"],
            "report": "PROBE 接近 1 支持整图 AI-generated；TruFor 虽在内衣区域高响应，但整图生成后局部问题为 Not applicable，不另写局部篡改。",
        },
    },
    "genimage__Wukong__217_wukong_image88__q95": {
        "call_order": ["probe", "trufor"],
        "first": {
            "region": "none",
            "visual": "草地上的比格犬，白色身体、棕色耳朵和黑色眼罩斑纹，背景是绿色草地和栅栏；毛发和草地纹理可见。",
            "anomalies": "狗头部的边缘较整齐，但没有足以单独判定的局部证据；先测量整图来源。",
            "target": "probe — 先回答整图是否 AI 生成；局部问题稍后补测。",
        },
        "second": {
            "region": "[166, 604, 510, 1000]",
            "visual": "同一张比格犬照片；狗位于画面右侧，草地和栅栏保持连续。",
            "anomalies": "PROBE 0.9999 接近 1，支持整图 AI-generated；但为检查头部/耳朵区域是否有集中局部响应，补测 TruFor。",
            "target": "trufor — 整图已倾向生成，补测局部以确认头部高响应是否只是整图生成的一部分。",
        },
        "reasoning": "PROBE 0.9999 接近 1，支持整图 AI-generated。TruFor 0.8075，异常图的高响应落在狗头部和耳朵，可靠性图在该区域中等偏高；但原图整体毛发、草地和栅栏的质感与整图生成一致，头部响应没有独立的局部内容替换边界。按协议，整图 AI-generated 后局部为 Not applicable，不把头部高响应另写成局部篡改；总体 Fake。",
        "verdict": {
            "verdict": "Fake", "whole_image_generation": "AI-generated",
            "local_manipulation": "Not applicable",
            "suspicious_regions": [],
            "certainty": "High",
            "primary_evidence": ["probe_dino_expert"],
            "report": "PROBE 接近 1 支持整图 AI-generated；TruFor 头部高响应属于整图生成内容，局部为 Not applicable，不另写篡改。",
        },
    },
    "genimage__Midjourney__229_midjourney_197__q95": {
        "call_order": ["probe", "trufor"],
        "first": {
            "region": "none",
            "visual": "黑色背景前的白色长毛犬，毛发蓬松，站姿端正；画面像棚拍，背景纯黑，狗身与背景对比强烈。",
            "anomalies": "狗的毛发边缘非常整齐，先测量整图来源；局部细节稍后补测。",
            "target": "probe — 先回答整图是否 AI 生成；纯黑背景和整齐毛发需要整图判别器。",
        },
        "second": {
            "region": "none",
            "visual": "同一张白色长毛犬照片；狗身占据画面中部，黑色背景纯色。",
            "anomalies": "PROBE 0.9886 高于阈值，支持整图 AI-generated；补测 TruFor 看狗身是否有局部响应，以及该响应是否可靠。",
            "target": "trufor — 整图已倾向生成；补测局部并重点读可靠性图，避免把不可靠的高响应当成独立篡改。",
        },
        "reasoning": "PROBE 0.9886 高于阈值，支持整图 AI-generated。TruFor 0.5149 仅略高于阈值，异常图的高响应覆盖狗身大部，但可靠性图在狗身区域明显偏低（蓝色/青色）；该局部响应不能作为独立篡改证据。按协议，整图 AI-generated 后局部为 Not applicable；总体由整图生成支撑为 Fake，报告注明局部图不可靠这一限制。",
        "verdict": {
            "verdict": "Fake", "whole_image_generation": "AI-generated",
            "local_manipulation": "Not applicable",
            "suspicious_regions": [],
            "certainty": "High",
            "primary_evidence": ["probe_dino_expert"],
            "report": "PROBE 支持整图 AI-generated；TruFor 的狗身响应可靠性低，不能当作独立局部篡改，局部为 Not applicable。",
        },
    },
    "genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95": {
        "call_order": ["probe", "trufor"],
        "first": {
            "region": "none",
            "visual": "狗脸特写，画面几乎被毛发和口鼻占满；毛发细节密集，眼睛和鼻头可见，背景虚化。",
            "anomalies": "毛发走向和口鼻边缘略显过度平滑，先测量整图来源；局部细节稍后补测。",
            "target": "probe — 先回答整图是否 AI 生成；纯色/虚化背景和过度平滑的毛发需要整图判别器。",
        },
        "second": {
            "region": "[170, 75, 675, 725]",
            "visual": "同一张狗脸特写；口鼻、眼睛和周围毛发保持连续，背景虚化。",
            "anomalies": "PROBE 0.9672 高于阈值，支持整图 AI-generated；补测 TruFor 检查口鼻/面部是否有局部响应，并读该响应的可靠性。",
            "target": "trufor — 整图已倾向生成，补测局部是为了确认口鼻区域的响应是否只是整图生成的一部分，以及可靠性是否支持任何局部结论。",
        },
        "reasoning": "PROBE 0.9672 高于阈值，支持整图 AI-generated；原图毛发整体过度平滑、口鼻边缘缺少自然碎毛，和整图生成一致。TruFor 0.1489 低于阈值，异常图在口鼻区域有宽而低的中等响应，可靠性图在同一区域偏低（低可靠占比较高），不能作为独立篡改证据。按协议，整图判 AI-generated 后局部问题为 Not applicable；局部图只用于确认没有可靠的独立局部证据，总体 Fake，certainty High。",
        "verdict": {
            "verdict": "Fake", "whole_image_generation": "AI-generated",
            "local_manipulation": "Not applicable",
            "suspicious_regions": [],
            "certainty": "High",
            "primary_evidence": ["probe_dino_expert"],
            "report": "PROBE 支持整图 AI-generated；TruFor 低于阈值且口鼻响应可靠性低，不能另写局部篡改，局部为 Not applicable。",
        },
    },
    "genimage__BigGAN__545_biggan_00074__q95": {
        "call_order": ["probe", "trufor"],
        "first": {
            "region": "none",
            "visual": "橙色背景前的金属/木质圆形器物，表面有放射状纹样和中心凸起；光线均匀，阴影柔和，背景纯色。",
            "anomalies": "器物的对称纹样和中心结构略显规则，先测量整图来源；局部细节稍后补测。",
            "target": "probe — 先回答整图是否 AI 生成；纯色背景和规则纹样需要整图判别器。",
        },
        "second": {
            "region": "none",
            "visual": "同一张圆形器物照片；器物居中，橙色背景纯色，阴影柔和。",
            "anomalies": "PROBE 0.9959 高于阈值，支持整图 AI-generated；补测 TruFor 检查器物表面是否有局部高响应。",
            "target": "trufor — 整图已倾向生成，补测局部以确认表面纹样是否产生可靠的局部响应。",
        },
        "reasoning": "PROBE 0.9959 高于阈值，支持整图 AI-generated；原图纯色背景、均匀光线和规则纹样与整图生成一致。TruFor 0.2474 低于阈值，异常图没有大面积高响应，可靠性图整体较高；器物表面的纹样响应很弱，不能支持独立的局部篡改。按协议，整图判 AI-generated 后局部问题为 Not applicable；总体 Fake，certainty High。",
        "verdict": {
            "verdict": "Fake", "whole_image_generation": "AI-generated",
            "local_manipulation": "Not applicable",
            "suspicious_regions": [],
            "certainty": "High",
            "primary_evidence": ["probe_dino_expert"],
            "report": "PROBE 支持整图 AI-generated；TruFor 低于阈值且无大面积高响应，不能另写局部篡改，局部为 Not applicable。",
        },
    },
    "genimage__ADM__921_adm_7__q95": {
        "call_order": ["probe", "trufor"],
        "first": {
            "region": "none",
            "visual": "泛黄旧书页的特写，页面上有密集的手写/印刷文字和装饰线条，纸张边缘磨损；文字整体可辨但局部排列不规整。",
            "anomalies": "文字的笔画和行距不完全自然，先测量整图来源；局部细节稍后补测。",
            "target": "probe — 先回答整图是否 AI 生成；旧书页的纹理和文字需要整图判别器。",
        },
        "second": {
            "region": "none",
            "visual": "同一张旧书页特写；纸张纹理、文字和边缘磨损保持连续。",
            "anomalies": "PROBE 0.9898 高于阈值，支持整图 AI-generated；补测 TruFor 检查书页文字区域是否有局部高响应。",
            "target": "trufor — 整图已倾向生成，补测局部以确认文字区域是否存在可靠的局部响应。",
        },
        "reasoning": "PROBE 0.9898 高于阈值，支持整图 AI-generated；原图旧书页的泛黄、磨损和文字排列整体带有生成痕迹，和整图生成一致。TruFor 0.1590 低于阈值，异常图没有大面积高响应，可靠性图整体较高；文字区域的响应很弱，不能支持独立的局部篡改。按协议，整图判 AI-generated 后局部问题为 Not applicable；总体 Fake，certainty High。",
        "verdict": {
            "verdict": "Fake", "whole_image_generation": "AI-generated",
            "local_manipulation": "Not applicable",
            "suspicious_regions": [],
            "certainty": "High",
            "primary_evidence": ["probe_dino_expert"],
            "report": "PROBE 支持整图 AI-generated；TruFor 低于阈值且无大面积高响应，不能另写局部篡改，局部为 Not applicable。",
        },
    },

    # ------------------------------------------------------------------
    # GenImage Real photographs
    # ------------------------------------------------------------------
    "genimage__Real__96d68646-b0a2-450e-ac0a-a5a1b58aa6c8__q95": {
        "call_order": ["probe", "trufor"],
        "first": {
            "region": "none",
            "visual": "黑色背景前的彩色网格球体，表面有粉色、紫色和蓝色的渐变，格纹清晰；光线形成自然高光，球体下方有阴影。",
            "anomalies": "球体上缘的高光较亮，但没有明显的局部拼接边；先测量整图来源。",
            "target": "probe — 先回答整图是否 AI 生成；再看是否需要局部测量支持 Not tampered。",
        },
        "second": {
            "region": "[183, 406, 270, 520]",
            "visual": "同一张彩色球体照片；球体、黑色背景和阴影保持连续。",
            "anomalies": "PROBE 0.0166 低于阈值，倾向 Real；补测 TruFor 检查球体上缘高光附近是否有局部高响应。",
            "target": "trufor — 整图已倾向 Real，补测局部以支持局部 Not tampered，并检查高光处是否有候选响应。",
        },
        "reasoning": "PROBE 0.0166 低于阈值，支持整图 Real。TruFor 0.0995 低于阈值，异常图只在球体上缘有一个小高响应，可靠性图在该处较高；原图球体、黑色背景和阴影连续，高光属于正常反射，没有内容替换边界。局部证据不足以支持 Tampered，按协议写 Not tampered；总体 Real，certainty High。",
        "verdict": {
            "verdict": "Real", "whole_image_generation": "Real",
            "local_manipulation": "Not tampered",
            "suspicious_regions": [],
            "certainty": "High",
            "primary_evidence": ["probe_dino_expert", "trufor_expert"],
            "report": "PROBE 与 TruFor 均低于阈值；球体上缘小响应属正常高光，原图内容连续，整图与局部均判 Real。",
        },
    },
    "genimage__Real__ac71afc5-10b2-4a79-86e2-aa55191c0d10__q95": {
        "call_order": ["probe", "trufor"],
        "first": {
            "region": "none",
            "visual": "森林中的一只手捏着松果，背景是虚化的树干和绿色光斑；手指、松果鳞片和树皮纹理可见。",
            "anomalies": "松果和手指的边缘较清楚，但没有明显拼接边；先测量整图来源。",
            "target": "probe — 先回答整图是否 AI 生成；PROBE 的整图分数可能与自然场景不一致，需要记录冲突。",
        },
        "second": {
            "region": "[643, 212, 1000, 590]",
            "visual": "同一张松果照片；手、松果和虚化森林保持连续。",
            "anomalies": "PROBE 0.5869 仅略高于阈值，倾向 AI-generated，但分数接近边界；为检查手/松果区域是否有局部响应，补测 TruFor。",
            "target": "trufor — PROBE 只是边界高分，且自然场景与其不一致；补测局部，看手和松果附近是否有高响应。",
        },
        "reasoning": "PROBE 0.5869 仅略高于阈值，是整图判别器的原始分数，不定位也不是校准概率；原图手指、松果鳞片和虚化森林保持自然的照片连续性，但没有校准依据把边界分数直接翻成 Real。TruFor 0.2037 低于阈值，异常图在手/松果边缘有轻微响应，可靠性图在该区域中等偏高；该响应没有清楚的内容替换边界。局部判 Not tampered，整图因 PROBE 边界高分保留 Uncertain，总体 Uncertain，certainty 为 null。",
        "verdict": {
            "verdict": "Uncertain", "whole_image_generation": "Uncertain",
            "local_manipulation": "Not tampered",
            "suspicious_regions": [],
            "certainty": None,
            "primary_evidence": ["probe_dino_expert", "trufor_expert"],
            "report": "TruFor 低分支持局部 Not tampered；PROBE 仅边界高分、与自然场景不一致，整图保留 Uncertain。",
        },
    },
    "genimage__Real__b3cfabea-3cae-414b-a9e7-37f421542552__q95": {
        "call_order": ["trufor", "probe"],
        "first": {
            "region": "none",
            "visual": "岩石间的瀑布，水流从岩壁落下，周围有绿色苔藓和灌木；水花、岩石纹理和阴影可见。",
            "anomalies": "瀑布边缘的水花较亮，但没有明显拼接边；先做局部测量。",
            "target": "trufor — 先检查水流和岩石附近是否有局部高响应，再补测整图来源。",
        },
        "second": {
            "region": "none",
            "visual": "同一张瀑布照片；水流、岩石、苔藓和灌木保持连续。",
            "anomalies": "TruFor 0.4589 低于阈值，异常图的小高响应出现在画面底部和右缘，未落在岩石或植被的内容边界上；可靠性图整体较高。整图来源尚未测量。",
            "target": "probe — 局部低于阈值且响应在边缘；补测 PROBE 确认整图来源，再完成局部 Not tampered 的判断。",
        },
        "reasoning": "TruFor 0.4589 低于阈值，异常图的小高响应只在画面底部和右缘，未落在岩石、苔藓或灌木的内容边界上；可靠性图整体较高。原图水流、岩石纹理和植被连续，没有内容替换证据，局部判 Not tampered。PROBE 0.0017 低于阈值，支持整图 Real。总体 Real，certainty High；报告保留“异常图边缘小响应”这一限制。",
        "verdict": {
            "verdict": "Real", "whole_image_generation": "Real",
            "local_manipulation": "Not tampered",
            "suspicious_regions": [],
            "certainty": "High",
            "primary_evidence": ["probe_dino_expert", "trufor_expert"],
            "report": "TruFor 低于阈值，边缘小响应未落在内容上；PROBE 低分支持整图 Real，局部 Not tampered。",
        },
    },
    "genimage__Real__e4ce4d06-00f7-456f-a17e-6049516d9668__q95": {
        "call_order": ["trufor", "probe"],
        "first": {
            "region": "[64, 128, 1000, 636]",
            "visual": "树叶间的狐猴，灰色身体、白色脸部和黄色眼睛，背景是绿色枝叶；毛发和叶片纹理可见。",
            "anomalies": "狐猴身体与树叶的交界较复杂，先做局部测量；画面没有文字或明显拼接边。",
            "target": "trufor — 先检查狐猴身体附近的异常响应，并重点看可靠性图。",
        },
        "second": {
            "region": "[64, 128, 1000, 636]",
            "visual": "同一张狐猴照片；狐猴、树叶和枝条保持连续。",
            "anomalies": "TruFor 0.3959 低于阈值，异常图在狐猴身体上有较宽的中等响应，但可靠性图在身体区域偏低（蓝色/青色），低可靠占比较高；该响应不能作为内容证据。整图来源尚未测量。",
            "target": "probe — 局部低于阈值但响应较宽且可靠性低；补测 PROBE 确认整图来源，避免把低可靠响应写成篡改。",
        },
        "reasoning": "TruFor 0.3959 低于阈值，异常图在狐猴身体上有较宽的中等响应，但可靠性图在身体区域明显偏低，低可靠占比较高；按协议，低可靠区域的响应不能当作篡改证据。原图狐猴毛发、树叶和枝条连续，没有内容替换边界，局部判 Not tampered。PROBE 0.0001 低于阈值，支持整图 Real。总体 Real，certainty Medium；主要限制是局部图可靠性低，但分数低于阈值且内容连续。",
        "verdict": {
            "verdict": "Real", "whole_image_generation": "Real",
            "local_manipulation": "Not tampered",
            "suspicious_regions": [],
            "certainty": "Medium",
            "primary_evidence": ["probe_dino_expert", "trufor_expert"],
            "report": "PROBE 低分支持整图 Real；TruFor 低于阈值，狐猴身上的宽响应可靠性低且无内容边界，局部 Not tampered。",
        },
    },

    # ------------------------------------------------------------------
    # Supplementary single-expert exemplars
    # ------------------------------------------------------------------
    "single_probe__genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95": {
        "call_order": ["probe"],
        "first": {
            "region": "none",
            "visual": "狗脸特写，画面几乎被毛发和口鼻占满；毛发细节密集，眼睛和鼻头可见，背景虚化。",
            "anomalies": "毛发走向和口鼻边缘略显过度平滑，先测量整图来源。",
            "target": "probe — 整图来源尚未测量；PROBE 是整图来源的判别器。",
        },
        "second": None,
        "reasoning": "PROBE 0.9672 高于阈值，支持整图 AI-generated；原图毛发整体过度平滑、口鼻边缘缺少自然碎毛，和整图生成一致。按协议，整图判 AI-generated 后局部问题为 Not applicable；TruFor 即使给出局部响应，也不会改变整图结论，因此不再补测，直接结案。总体 Fake，certainty High。",
        "verdict": {
            "verdict": "Fake", "whole_image_generation": "AI-generated",
            "local_manipulation": "Not applicable",
            "suspicious_regions": [],
            "certainty": "High",
            "primary_evidence": ["probe_dino_expert"],
            "report": "PROBE 0.9672 支持整图 AI-generated，毛发整体过度平滑；局部问题 Not applicable，单专家直接结案。",
        },
    },
    "single_trufor__autosplice__54029__1__q95": {
        "call_order": ["trufor"],
        "first": {
            "region": "[300, 760, 935, 910]",
            "visual": "户外树荫下的晾衣绳，挂着多件衣物；右侧一件白色衣物表面很亮，下缘与背景的分界较直。",
            "anomalies": "右侧白色衣物的亮度和边界与周围柔软布料不一致，先做局部测量。",
            "target": "trufor — 右侧白色衣物是主要局部候选，先确认异常响应和可靠性。",
        },
        "second": None,
        "reasoning": "TruFor 0.9995，异常图的高响应落在右侧白色衣物，可靠性图在同一区域为高可靠，原图该处内容边界较直；局部证据支持 Tampered。整图来源尚未测量，PROBE 没有调用，因此不能把整图判成 Real 或 AI-generated，整图保留 Uncertain。局部 Tampered 足以支撑总体 Fake；certainty 取 Medium，因为整图来源仍未解决。",
        "verdict": {
            "verdict": "Fake", "whole_image_generation": "Uncertain",
            "local_manipulation": "Tampered",
            "suspicious_regions": [{
                "bbox": [300, 760, 935, 910],
                "reason": "TruFor 高响应集中在右侧白色衣物，可靠性图同区域可靠，原图该处内容边界较直",
                "evidence": ["{{TRUFOR_ID}}"],
            }],
            "certainty": "Medium",
            "primary_evidence": ["trufor_expert"],
            "report": "TruFor 高响应集中在右侧白色衣物且可靠，局部 Tampered 支撑总体 Fake；整图来源未测，保留 Uncertain。",
        },
    },
}
