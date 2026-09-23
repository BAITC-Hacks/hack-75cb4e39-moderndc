# HackAlem — «Граф денег», этап 1

Детерминированное аналитическое ядро на основе предоставленного `starter.py`.
Роли описывают наблюдаемую структуру переводов. `priority_score` — приоритет
дальнейшей AML-проверки, **не вероятность преступления и не доказательство нарушения**.

Запуск из корня репозитория (обычный Python, без `-O`, поскольку проверки используют assert):

```powershell
python -m pip install -r requirements.txt
python starter.py --data data --out out
python -m unittest -v test_analytics
```

Результаты: `out/nodes_roles.csv`, `out/clusters.csv`, `out/top_nodes.csv`.
Стартовый загрузчик, `sanity_check`, построение графа и базовые признаки сохранены
и расширены. Формулы, кластеризация и проверка выгрузок находятся в `analytics.py`.

## Фактические входные данные

| Файл | Строк | Схема parquet |
|---|---:|---|
| nodes | 2248 | gid int64, depth int64, is_seed bool |
| edges | 3119 | src/dst int64, sum_kzt float64, n_tx int64, depth int8 |
| transactions | 4840 | src/dst int64, date string/object, sum_kzt float64 |

Размеры parquet: nodes 11905 байт, edges 34187 байт, transactions 40745 байт.

Пропусков, дубликатов gid, повторных пар рёбер, неизвестных концов рёбер и петель
в предоставленных данных нет. Gid лежат между 100000000011452100 и
100000008782800100: они везде сохраняются как целые, без преобразования в float.
Канонический порядок — числовой порядок int64 gid.

81 seed, включая 19 узлов без рёбер. Распределение depth 0..4: 81, 472, 462,
789, 444. Все 444 узла depth=4 не имеют наблюдаемого выхода.
Рёберные суммы: 5000..4400000 KZT; n_tx: 1..67. Суммы отдельных переводов:
5000..3000000 KZT. Общая сумма: 365890012.01 KZT.
`sanity_check` проверяет пары прежним outer merge и дополнительно сравнивает
суммы (rtol=1e-12, atol=1e-6 KZT) и точные количества переводов.

Период: 2026-07-01..2026-07-31, 31 уникальная дата, времени суток нет.
`active_days` — число различных дней с любым входом или выходом;
`activity_span_days` — включительный интервал между первым и последним днём.
У isolates оба значения равны 0. Диапазон active_days на этих данных: 0..29.
`out_after_prior_day_in_tx` считает исходящие операции, для которых существует
хотя бы один вход **строго на предыдущий календарный день**;
`prior_day_in_fraction` делит это число на out_tx (при отсутствии выхода — 0).
Таких исходящих операций 1002. Совпадения в тот же день не учитываются;
будущие входы не используются. Это временное соседство, не установление источника денег.

## Граф и нормализация

Все gid добавляются до рёбер в DiGraph. Степени, объёмы, число переводов,
PageRank и betweenness направленные. PageRank использует sum_kzt как силу связи
(стандартный damping=0.85, tol=1e-12, max_iter=1000). Если не сходится —
печатается FALLBACK, возвращается равномерный вектор, его percentile равен 0.
Betweenness — точный, направленный, **без distance-weight**, вычисляется один раз.
KZT нигде не используется как расстояние. HITS не требуется.

Для неотрицательного признака P(x): нулевые значения получают 0;
положительные — `(average_rank - 0.5) / число положительных`.
Если вся колонка постоянна, всем 0: она не даёт различающего сигнала.
NaN/inf/отрицательные значения в нормализуемых признаках отвергаются.

`pass_through = out_kzt / in_kzt` определён только при in_kzt>0.
Во время inference неопределённость сохраняется с отдельной маской
`pass_through_defined`. Только после inference в CSV записывается sentinel **-1**,
который не означает реальный ratio. Для seed этот ratio не участвует в scores.

## Фиксированные формулы ролей

Обозначения: I = среднее P(in_deg), P(in_kzt), P(in_tx);
O — аналогичное среднее исходящих признаков; B=P(betweenness), R=P(pagerank);
T=prior_day_in_fraction. Si=in_deg/max(in_deg+out_deg,1), So аналогично.
Q=min(out/in,in/out) используется только для non-seed с положительным входом
и выходом; это близость наблюдаемых объёмов, не прослеживание отдельных денег.

| Роль | Score и условие |
|---|---|
| consolidator | I·(0.5+0.5·Si), если in_deg≥2; иначе 0 |
| distributor | O·(0.5+0.5·So), если out_deg≥2; иначе 0 |
| transit, non-seed | min(I,O)·(0.65+0.25·Q+0.10·T), только при наличии входа и выхода |
| transit, seed | min(I,O)·(0.90+0.10·T), только при наличии входа и выхода |
| terminal, non-seed | I·clip(1−out/in,0,1), только при in_deg>0 и отсутствии truncation |
| terminal, seed | I, только при in_deg>0, out_deg=0 и отсутствии truncation |
| coordinator | 0.70·B+0.30·R, только при входе, выходе и betweenness>0 |
| peripheral | 1−max(scores остальных пяти ролей) |

Входящий/исходящий degree даёт структурное приближение authority/hub без
добавления отдельного нестабильного спектрального алгоритма.
Данные censored на четвёртом колене: обход здесь заканчивается, и у всех 444
depth=4 узлов out_deg=0. Отсутствие наблюдаемого исходящего ребра не доказывает
retention (остановку денег). Поэтому classifier не использует это отсутствие
как terminal evidence: при truncated_by_depth score_terminal=0.
Terminal/non-terminal status по отсутствию выхода неидентифицируем из этой
censored выгрузки. Эти узлы не объявляются доказанно non-terminal: реальный
downstream по предоставленным данным неизвестен и не восстанавливается.
Seed обрабатываются отдельной формулой из-за неполного
наблюдаемого входа. Для isolated node все основные scores равны 0, peripheral=1.

Scores округляются до 10 знаков, выбирается максимум. Явный порядок равенств:
consolidator, transit, distributor, terminal, coordinator, peripheral.
Все шесть scores сохранены в диагностических колонках. Коэффициенты едины
для всех узлов и не калибровались под желаемое распределение ролей.

## Кластеры и приоритет

Только для Louvain строится отдельная неориентированная проекция: веса
противоположных направлений складываются. Seed алгоритма 42, resolution=1.
Isolates получают отдельные communities. Узлы внутри community сортируются;
communities сортируются по минимальному gid, затем получают номера 0..K−1.
Все узлы/рёбра поступают в алгоритмы в каноническом порядке.

Внутренний объём кластера считается по исходным направленным рёбрам, каждое
ровно один раз. `top_gids` — JSON-массив до 5 gid по убыванию betweenness,
затем PageRank, затем возрастанию gid (метрики округлены до 10 знаков).
`hypothesis` — структурно-функциональная гипотеза о назначении community,
основанная только на уже назначенных ролях и наблюдаемых потоках.
`dominant_role` — роль с максимальным count; равенства разрешаются строго по
ROLES order: consolidator, transit, distributor, terminal, coordinator, peripheral.
`dominant_share` = dominant count / n_nodes, формат с 10 знаками после точки.
Соответствие dominant role → category:

| Роль | Category |
|---|---|
| consolidator | collection-oriented |
| transit | transit-oriented |
| distributor | distribution-oriented |
| terminal | terminal/retention-oriented |
| coordinator | coordination/bridging-oriented |
| peripheral | peripheral/isolated |

Singleton с in_deg=out_deg=0 получает category=peripheral/isolated.
При этом правиле ориентация всегда определена; mixed и дополнительные пороги
не используются. Строка также содержит counts всех шести ролей (включая нулевые)
в ROLES order, internal KZT с 2 знаками после точки и truncated count.
Фиксированные форматирование и tie-break обеспечивают независимость от порядка
строк. Окончание `structural hypothesis only` подчёркивает, что это гипотеза,
а не вывод о преступной деятельности или доказанной незаконности community.

V=P(in_kzt+out_kzt), F=P(in_tx+out_tx).
`priority = 0.35·B + 0.15·R + 0.25·V + 0.15·F + 0.10·T·F`.
Интерпретация: посредничество и положение в потоке, наблюдаемый объём,
частота и временное соседство с поправкой на частоту.
Назначенные role/role_score не входят в формулу; для isolates priority=0.
Большие объёмы и высокая частота учитываются отдельно. Это эвристика без
разметки, её значения не калиброваны как вероятности. Ограничения выборки
сохраняются: seed-вход неполон, depth=4 обрезан.

Top N=30, порядок: округлённый priority убывает, gid возрастает.
`why` содержит численные компоненты формулы. `evidence` содержит степени,
объёмы, числа переводов, сигнал выбранной роли и признаки seed/truncation.

## CHECK → FIX → RECHECK

До записи и после чтения CSV проверяются universe gid, обязательные значения,
численная безопасность всех диагностик, шесть scores и argmax, длина и числа
в evidence, полнота кластеров, n_nodes/n_seed, независимый пересчёт внутренних
сумм, состав top_gids, последовательные ranks, сортировка и совпадение top scores.
Отдельно проверяются orphan nodes, запрет terminal при truncation и маска ratio.

9 тестов проверяют также направленные агрегаты против parquet, weight=None
для betweenness, временной сигнал независимым подсчётом и примером с прошлым/
тем же/следующим днём, неизменность всех role scores при изменении ratio seed
и undefined, PageRank fallback, ошибочные суммы/числа переводов и повторяемость.
Тест hypothesis проверяет категории доминирующих ролей, singleton isolate,
все попарные равенства максимумов, долю, численные evidence и неизменность строки
при повторном вызове и перестановке узлов кластера. Validation проверяет hypothesis
каждого кластера до записи и после чтения CSV.
Повторный расчёт с переставленными строками всех трёх входов даёт идентичные
таблицы и SHA-256 сериализованных CSV. Выходная точность 10 знаков, LF.

Фактический результат: 2248 узлов, 88 кластеров, 30 top nodes; все 19 isolates
сохранены; evidence до 134 символов; все 444 truncated узла не классифицированы
как terminal из-за отсутствия независимого downstream evidence, что не доказывает
их фактический non-terminal status.
Роли: peripheral 1431, consolidator 248, terminal 247, coordinator 198,
distributor 111, transit 13. Эти количества — результат, не цель настройки.

Проверенное окружение: Python 3.14.7, pandas 3.0.6, numpy 2.3.5,
networkx 3.7, pyarrow 25.0.1, scipy 1.18.1; версии закреплены в requirements.txt.
Детерминизм гарантируется для одинаковых входов, параметров и окружения;
между версиями Louvain/численных библиотек результат может различаться.

## Этап 2 — локальный presentation / inspection UI

Streamlit 1.64.0 и Plotly 7.1.0 показывают готовый результат этапа 1.
`app.py` — интерфейс и локальная визуализация; `agent.py` — чтение, проверка
минимальной схемы и пять детерминированных действий; `test_agent.py` — проверки
источников данных и UI smoke tests. Формулы, роли и кластеры не пересчитываются.

Установка и запуск из корня репозитория:

```powershell
python -m pip install -r requirements.txt
python starter.py --data data --out out
python -m streamlit run app.py --server.address 127.0.0.1
```

Открыть локальный адрес, напечатанный Streamlit (обычно http://127.0.0.1:8501).
Пути данных определяются относительно `app.py`. Чтение кэшируется через
`st.cache_data`; изменение размера/mtime файла обновляет кэш. Отсутствующие
outputs показывают команду запуска analytics, malformed data — понятную ошибку.
UI не запускает analytics автоматически. Это локальное приложение без deployment.

- **Dashboard:** фактические counts, распределение ролей, готовый shortlist и
  гипотезы кластеров. Transactions = сумма n_tx из edges.parquet; отдельные
  transactions.parquet UI не загружает. Gid из shortlist можно скопировать в Inspector.
- **Node Inspector:** выбор из всех nodes_roles.csv или точный gid, сохранённые
  scores/evidence, доступные диагностические колонки, ограничения и cluster hypothesis.
  Основное объяснение — строка explanation из explain_node(gid); evidence входит
  в неё без изменения и отдельно повторно не выводится. Warnings показаны отдельно.
  Gid передаётся браузеру строкой; float и scientific notation не принимаются.
  Неопределённый pass_through показан как undefined, а не как реальный нулевой ratio.
- **Neighbors / local graph:** только выбранный gid и непосредственные соседи
  из edges.parquet. Каждая линия — фактическое incident directed edge со стрелкой
  и tooltip суммы/числа переводов; встречные направления разделены кривыми.
  Navy marker выделяет выбранный узел, green — incoming, amber — outgoing.
  Геометрическое размещение не имеет аналитического смысла. Полный граф по
  умолчанию не строится, скрытые downstream edges не добавляются. Isolates имеют empty state.
- **Cluster Inspector:** сохранённые counts, internal KZT, top_gids и hypothesis;
  members берутся только по cluster_id, сортируются по priority descending / gid ascending.
- **Agentic Analyst:** явный выбор inspect_node(gid), inspect_cluster(cluster_id),
  get_top_priority(n), get_neighbors(gid), explain_node(gid). Каждое действие
  проверяет параметр и наличие записи/допустимый диапазон, возвращает сохранённые
  данные. Dispatcher возвращает structured result и audit trace фактических операций.
  explain_node(gid) детерминированно формирует краткое presentation explanation
  из сохранённых gid, role, role_score, evidence и priority_score, с форматированием
  обоих scores до 4 знаков и пояснением смысла review priority. Seed/truncation
  limitations возвращаются отдельно, в порядке seed, затем truncation.
  Это не chatbot/LLM; natural-language intent recognition отсутствует.
  Для n разрешён только диапазон готового shortlist: новые места рейтинга не создаются.

Priority — приоритет проверки, не вероятность/доказательство нарушения.
Seed-вход может быть неполон; при depth=4 downstream censored и неизвестен,
отсутствие выхода не доказывает остановку денег или фактический non-terminal status.
Explain показывает оба предупреждения, если оба флага true. Временные признаки
имеют дневную точность; совпадение активности не доказывает источник финансирования.
Роли и гипотезы характеризуют только наблюдаемую структуру.

Проверки этапов по порядку:

```powershell
python starter.py --data data --out out
python -m unittest -v test_analytics
python -m unittest -v test_agent.AgentTests
python -m unittest -v test_agent.UISmokeTests
```

Smoke использует штатный [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest):
dashboard, выбор узла/кластера, agent action, seed/truncation warnings,
isolates, invalid inputs, missing/malformed state. Геометрия локального графа,
направленные стрелки и данные tooltips проверяются непосредственно в Plotly figure;
пиксельное browser testing не заявляется. Стрелки используют
[Plotly annotations](https://plotly.com/python/reference/layout/annotations/).
