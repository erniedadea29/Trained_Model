-- Run once in Supabase Dashboard -> SQL Editor -> New query -> Run

-- 1) Registry of trained model versions
create table if not exists ml_models (
  id           bigint generated always as identity primary key,
  name         text not null,
  algorithm    text not null,
  params       jsonb,
  metrics      jsonb,
  storage_path text not null,
  is_active    boolean not null default false,
  created_at   timestamptz not null default now()
);

-- 2) Saved predictions (one row per line item)
create table if not exists budget_predictions (
  id              bigint generated always as identity primary key,
  model_id        bigint references ml_models(id),
  event_title     text,
  event_date      date,
  semester        text,
  college         text,
  category        text,
  item            text,
  qty             int,
  pred_unit_cost  numeric(12,2),
  pred_subtotal   numeric(12,2),
  created_at      timestamptz not null default now()
);

-- 3) Private storage bucket for the .h5 file
insert into storage.buckets (id, name, public)
values ('models', 'models', false)
on conflict (id) do nothing;

-- 4) Row Level Security: the Python API uses the service_role key (bypasses RLS).
--    Logged-in app users may only READ predictions and model info.
alter table ml_models enable row level security;
alter table budget_predictions enable row level security;

drop policy if exists "read models" on ml_models;
create policy "read models" on ml_models for select to authenticated using (true);

drop policy if exists "read predictions" on budget_predictions;
create policy "read predictions" on budget_predictions for select to authenticated using (true);
