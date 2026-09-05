using Printf
using Test

# Run each function in `funcs` under every backend listed in `platforms`.
# A platform is skipped when its GPU package is not loadable or when
# `set_backend` reports it as unavailable (e.g. CUDA installed on a
# CPU-only CI runner), so the same suite covers CPU on CI and GPU locally.
# The global backend is restored to CPU afterwards.
function test_functions(test_name, funcs...;
                        platforms=["CPU", "CUDA", "AMDGPU", "oneAPI", "Metal"])
    for platform in platforms
        if platform != "CPU"
            if Base.find_package(platform) === nothing
                continue
            end
            try
                Base.eval(Main, :(using $(Symbol(platform))))
            catch
                continue
            end
        end

        if !set_backend(platform)
            continue
        end

        name = @sprintf("%s %s", test_name, platform)
        @testset "$name" begin
            for func in funcs
                # the GPU package may have been loaded by the `Base.eval`
                # above, whose new methods are not visible to code running
                # in the current world age; invokelatest re-enters at the
                # latest world so freshly loaded backends are picked up
                Base.invokelatest(func)
            end
        end
    end
    set_backend("cpu")
    return nothing
end
