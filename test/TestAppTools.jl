@testitem "AppTools - CLIArg" begin
    import FiniteElementContainers.AppTools as AT

    # default simple behavior
    arg = AT.CLIArg("--input-file")
    @test arg.has_default == false
    @test arg.has_input == true
    @test arg.has_short_name == false
    @test arg.is_required == true
    @test arg.default == ""
    @test arg.help_message == ""
    @test arg.name == "--input-file"
    @test arg.short_name == ""

    # next simplest with short name
    arg = AT.CLIArg("--input-file"; short_name = "-i")
    @test arg.has_default == false
    @test arg.has_input == true
    @test arg.has_short_name == true
    @test arg.is_required == true
    @test arg.default == ""
    @test arg.help_message == ""
    @test arg.name == "--input-file"
    @test arg.short_name == "-i"

    # next simplest with short name and default, not required
    arg = AT.CLIArg(
        "--input-file";
        default = "input-file.toml",
        help_message = "name of input file <input-file>.toml",
        is_required = false,
        short_name = "-i"
    )
    @test arg.has_default == true
    @test arg.has_input == true
    @test arg.has_short_name == true
    @test arg.is_required == false
    @test arg.default == "input-file.toml"
    @test arg.help_message == "name of input file <input-file>.toml"
    @test arg.name == "--input-file"
    @test arg.short_name == "-i"
end

@testitem "AppTools - CLIArgParser" begin
    import FiniteElementContainers.AppTools as AT

    # default with no other args
    args = [
        "--input-file", "input-file.toml",
        "--log-file", "log.log"
    ]
    parser = AT.CLIArgParser()
    AT.parse!(parser, args)
    println(parser)
    @test AT.get_cli_arg(parser, "--input-file") == "input-file.toml"
    @test AT.get_cli_arg(parser, "--log-file") == "log.log"


    # case where we added extra stuff
    args = [
        "--input-file", "input-file.toml",
        "--log-file", "log.log",
        "--backend", "cpu",
        "--verbose"
    ]
    parser = AT.CLIArgParser()
    AT.add_cli_arg!(parser, "--backend"; short_name = "-b")
    AT.add_cli_arg!(parser, "--verbose"; has_input = false, short_name = "-b")
    AT.parse!(parser, args)
    println(parser)
    @test AT.get_cli_arg(parser, "--input-file") == "input-file.toml"
    @test AT.get_cli_arg(parser, "--log-file") == "log.log"
    @test AT.get_cli_arg(parser, "--backend") == "cpu"
    @test AT.get_cli_arg(parser, "--verbose") == "true"

    # help message case long name
    args = [
        "--input-file", "input-file.toml",
        "--log-file", "log.log",
        "--backend", "cpu",
        "--verbose",
        "--help"
    ]
    @test_throws AssertionError AT.parse!(parser, args)

    # help message case short name
    args = [
        "--input-file", "input-file.toml",
        "--log-file", "log.log",
        "--backend", "cpu",
        "--verbose",
        "--h"
    ]
    @test_throws AssertionError AT.parse!(parser, args)

    # test missing option that is required
    args = [
        "--input-file", "input-file.toml",
        "--log-file", "log.log",
        "--verbose",
        "--h"
    ]
    @test_throws AssertionError AT.parse!(parser, args)
end

@testitem "AppTools - SimpleApp" begin
    import FiniteElementContainers.AppTools as AT
    args = [
        "--input-file", "input-file.toml",
        "--log-file", "log.log"
    ]
    app = AT.App("MyApp")
    # sim = AT.setup(app, args)
end
