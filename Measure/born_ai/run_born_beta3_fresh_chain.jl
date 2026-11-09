# Execution lifecycle only: the immutable runner still performs the whole chain.
include("run_born_scaling_fused.jl")

function fresh_chain_main()
    path=abspath(arg("plan",""))
    isfile(path) || error("provide frozen --plan")
    plan=validate_fused_plan(path)
    get(plan,"execution_mode","")=="fresh_process_per_chain" || error("wrong execution lifecycle")
    plan["source_sha256"][rel(@__FILE__)]==sha(@__FILE__) || error("entry point hash mismatch")
    run_id=arg("run-id","")
    isempty(run_id) && error("provide one --run-id from the frozen plan")
    job=only(filter(j->j["run_id"]==run_id,plan["jobs"]))
    run_job(job,plan,path)
end

fresh_chain_main()
