using Printf
using KernelAbstractions
using Test

# Run each function in `funcs` under every backend listed in `platforms`.
# A platform is skipped when its GPU package is not installed, not loadable,
# or has no functional device (e.g. CUDA installed on a CPU-only CI runner),
# so the same suite covers CPU on CI and GPU locally. The package keeps no
# backend state: every construction receives its backend explicitly (f(B)).
function test_functions(test_name, funcs...;
                        platforms=["CPU", "CUDA", "AMDGPU", "oneAPI", "Metal"])
    for platform in platforms
        B = _test_backend(platform)
        B === nothing && continue
        name = @sprintf("%s %s", test_name, platform)
        @testset "$name" begin
            for func in funcs
                # the GPU package may have been loaded by the `Base.eval`
                # above, whose new methods are not visible to code running
                # in the current world age; invokelatest re-enters at the
                # latest world so freshly loaded backends are picked up
                Base.invokelatest(func, B)
            end
        end
    end
    return nothing
end

# the backend for a platform name: CPU() for "CPU", nothing when the vendor
# package is missing/not loadable/not functional (platform skipped)
function _test_backend(platform)
    platform == "CPU" && return KernelAbstractions.CPU()
    Base.find_package(platform) === nothing && return nothing
    try
        Base.eval(Main, :(using $(Symbol(platform))))
    catch
        return nothing
    end
    # eval-loaded methods do not advance this task's world age
    return Base.invokelatest() do
        try
            HMatrixGPU.backend_from_name(lowercase(platform))
        catch
            nothing          # not functional here (e.g. GPU-less CI) → skip
        end
    end
end
