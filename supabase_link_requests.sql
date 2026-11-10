-- Run once in Supabase Dashboard -> SQL Editor (after supabase_setup.sql).
-- Links predictions to your existing budget requests. Only budget_predictions is changed;
-- budget_requests and budget_request_items are NOT modified.

alter table budget_predictions
  add column if not exists request_id        uuid references budget_requests(id) on delete cascade,
  add column if not exists request_item_id   uuid references budget_request_items(id) on delete cascade,
  add column if not exists requested_amount  numeric(12,2),
  add column if not exists variance_amount   numeric(12,2),   -- requested - predicted
  add column if not exists variance_pct      numeric(8,2),    -- (requested - predicted) / predicted * 100
  add column if not exists flag              text,            -- 'OVER' | 'UNDER' | 'OK'
  add column if not exists model_category    text,            -- category name the model used
  add column if not exists low_confidence    boolean default false;  -- category not in training data

create index if not exists budget_predictions_request_id_idx on budget_predictions(request_id);
create unique index if not exists budget_predictions_request_item_uidx
  on budget_predictions(request_item_id) where request_item_id is not null;

-- One row per budget request: requested vs predicted totals (handy for the adviser/dean screens)
create or replace view budget_request_predictions
with (security_invoker = true) as
select
  p.request_id,
  r.title,
  r.department_code,
  r.status,
  sum(p.requested_amount)                         as requested_total,
  sum(p.pred_subtotal)                            as predicted_total,
  sum(p.requested_amount) - sum(p.pred_subtotal)  as variance_amount,
  round((sum(p.requested_amount) - sum(p.pred_subtotal)) / nullif(sum(p.pred_subtotal), 0) * 100, 2) as variance_pct,
  count(*) filter (where p.flag = 'OVER')         as items_over,
  bool_or(p.low_confidence)                       as has_low_confidence_items,
  max(p.created_at)                               as predicted_at,
  max(p.model_id)                                 as model_id
from budget_predictions p
join budget_requests r on r.id = p.request_id
group by p.request_id, r.title, r.department_code, r.status;
