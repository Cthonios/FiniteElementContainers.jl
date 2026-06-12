include("SimpleYAML.jl")
using .SimpleYAML
using Test

# ── helpers ────────────────────────────────────────────────────────────────────
pass = 0
fail = 0

# macro test(expr)
#     quote
#         try
#             result = $(esc(expr))
#             if result === true
#                 global pass += 1
#             else
#                 global fail += 1
#                 println("FAIL: ", $(string(expr)), "  => ", result)
#             end
#         catch e
#             global fail += 1
#             println("ERROR: ", $(string(expr)), "  => ", e)
#         end
#     end
# end

# ── scalars ───────────────────────────────────────────────────────────────────
println("=== Scalars ===")
@test is_null(SimpleYAML.loads("null"))
@test is_null(SimpleYAML.loads("~"))
@test is_null(SimpleYAML.loads("Null"))
@test as_bool(SimpleYAML.loads("true"))  == true
@test as_bool(SimpleYAML.loads("false")) == false
@test as_bool(SimpleYAML.loads("True"))  == true
@test as_bool(SimpleYAML.loads("FALSE")) == false
@test as_int(SimpleYAML.loads("42"))   == 42
@test as_int(SimpleYAML.loads("-7"))   == -7
@test as_int(SimpleYAML.loads("0"))    == 0
@test as_int(SimpleYAML.loads("0xff")) == 255
@test as_int(SimpleYAML.loads("0o17")) == 15
@test as_int(SimpleYAML.loads("0b1010")) == 10
@test as_float(SimpleYAML.loads("3.14"))  ≈ 3.14
@test as_float(SimpleYAML.loads("-1.5e2")) ≈ -150.0
@test isinf(as_float(SimpleYAML.loads(".inf")))
@test isinf(as_float(SimpleYAML.loads("-.inf")))
@test isnan(as_float(SimpleYAML.loads(".nan")))
@test as_string(SimpleYAML.loads("hello")) == "hello"
@test as_string(SimpleYAML.loads("hello world")) == "hello world"

# ── quoted strings ─────────────────────────────────────────────────────────────
println("=== Quoted strings ===")
@test as_string(SimpleYAML.loads("\"hello\"")) == "hello"
@test as_string(SimpleYAML.loads("'hello'")) == "hello"
@test as_string(SimpleYAML.loads("\"tab\\there\"")) == "tab\there"
@test as_string(SimpleYAML.loads("\"new\\nline\"")) == "new\nline"
@test as_string(SimpleYAML.loads("\"quote\\\"here\"")) == "quote\"here"
@test as_string(SimpleYAML.loads("'it''s fine'")) == "it's fine"
@test as_string(SimpleYAML.loads("\"unicode \\u0041\"")) == "unicode A"
@test as_string(SimpleYAML.loads("\"null\"")) == "null"   # quoted null is a string
@test as_string(SimpleYAML.loads("\"true\"")) == "true"   # quoted bool is a string
@test as_string(SimpleYAML.loads("\"42\""))   == "42"     # quoted int is a string

# ── simple mapping ─────────────────────────────────────────────────────────────
println("=== Simple mapping ===")
v = SimpleYAML.loads("""
name: Alice
age: 30
active: true
score: 9.5
""")
d = SimpleYAML.as_dict(v)
@test as_string(d["name"])  == "Alice"
@test as_int(d["age"])      == 30
@test as_bool(d["active"])  == true
@test as_float(d["score"])  ≈ 9.5

# ── nested mapping ─────────────────────────────────────────────────────────────
println("=== Nested mapping ===")
v = SimpleYAML.loads("""
server:
  host: localhost
  port: 8080
  tls: false
""")
d = SimpleYAML.as_dict(SimpleYAML.as_dict(v)["server"])
@test as_string(d["host"]) == "localhost"
@test as_int(d["port"]) == 8080
@test as_bool(d["tls"]) == false

# ── block sequence ─────────────────────────────────────────────────────────────
println("=== Block sequence ===")
v = SimpleYAML.loads("""
- 1
- 2
- 3
""")
a = SimpleYAML.as_array(v)
@test length(a) == 3
@test as_int(a[1]) == 1
@test as_int(a[2]) == 2
@test as_int(a[3]) == 3

# ── sequence of mappings ───────────────────────────────────────────────────────
println("=== Sequence of mappings ===")
v = SimpleYAML.loads("""
- name: Bob
  age: 25
- name: Carol
  age: 28
""")
a = SimpleYAML.as_array(v)
@test length(a) == 2
@test as_string(as_dict(a[1])["name"]) == "Bob"
@test as_int(as_dict(a[2])["age"]) == 28

# ── mapping with sequence value ────────────────────────────────────────────────
println("=== Mapping with sequence value ===")
v = SimpleYAML.loads("""
colors:
  - red
  - green
  - blue
""")
arr = SimpleYAML.as_array(SimpleYAML.as_dict(v)["colors"])
@test length(arr) == 3
@test as_string(arr[1]) == "red"
@test as_string(arr[3]) == "blue"

# ── flow sequences ─────────────────────────────────────────────────────────────
println("=== Flow sequences ===")
v = SimpleYAML.loads("[1, 2, 3]")
a = as_array(v)
@test length(a) == 3
@test as_int(a[2]) == 2

v = SimpleYAML.loads("[\"a\", 'b', c]")
a = as_array(v)
@test as_string(a[1]) == "a"
@test as_string(a[2]) == "b"
@test as_string(a[3]) == "c"

# ── flow mappings ──────────────────────────────────────────────────────────────
# println("=== Flow mappings ===")
# v = SimpleYAML.loads("{x: 1, y: 2}")
# d = as_dict(v)
# @show d
# @test as_int(d["x"]) == 1
# @test as_int(d["y"]) == 2

# ── inline flow in block context ───────────────────────────────────────────────
# println("=== Inline flow in block ===")
# v = SimpleYAML.loads("""
# point: {x: 10, y: 20}
# tags: [a, b, c]
# """)
# d = as_dict(v)
# @test as_int(as_dict(d["point"])["x"]) == 10
# @test as_string(as_array(d["tags"])[2]) == "b"

# ── comments ──────────────────────────────────────────────────────────────────
println("=== Comments ===")
v = SimpleYAML.loads("""
# top comment
name: Dave  # inline comment
# mid comment
age: 40
""")
d = as_dict(v)
@test as_string(d["name"]) == "Dave"
@test as_int(d["age"]) == 40

# ── literal block scalar (|) ──────────────────────────────────────────────────
println("=== Literal block scalar ===")
v = SimpleYAML.loads("""
text: |
  Hello
  World
""")
s = as_string(as_dict(v)["text"])
@test s == "Hello\nWorld\n"

v = SimpleYAML.loads("""
text: |-
  Hello
  World
""")
@test as_string(as_dict(v)["text"]) == "Hello\nWorld"

# ── folded block scalar (>) ───────────────────────────────────────────────────
println("=== Folded block scalar ===")
v = SimpleYAML.loads("""
text: >
  Hello
  World
""")
s = as_string(as_dict(v)["text"])
@test s == "Hello World\n"

# ── anchors and aliases ────────────────────────────────────────────────────────
println("=== Anchors and aliases ===")
v = SimpleYAML.loads("""
defaults: &defs
  timeout: 30
  retries: 3

production:
  <<: *defs
  host: prod.example.com
""")
# Just check anchors were stored (merge key '<<' is not auto-applied, but alias is resolved)
d = as_dict(v)
@test as_int(as_dict(d["defaults"])["timeout"]) == 30

# Simple alias usage
v = SimpleYAML.loads("""
base: &b 42
copy: *b
""")
d = as_dict(v)
@test as_int(d["base"]) == 42
@test as_int(d["copy"]) == 42

# ── null value ────────────────────────────────────────────────────────────────
println("=== Null value ===")
v = SimpleYAML.loads("""
key1: null
key2: ~
key3:
""")
d = as_dict(v)
@test is_null(d["key1"])
@test is_null(d["key2"])
@test is_null(d["key3"])

# ── deeply nested ─────────────────────────────────────────────────────────────
println("=== Deeply nested ===")
v = SimpleYAML.loads("""
simulation:
  mesh:
    elements: 1024
    order: 2
  material:
    density: 7800.0
    moduli:
      - 210e9
      - 0.3
  boundary_conditions:
    - type: fixed
      nodes: [1, 2, 3]
    - type: load
      value: -1000.0
""")
sim = as_dict(as_dict(v)["simulation"])
@test as_int(as_dict(sim["mesh"])["elements"]) == 1024
@test as_float(as_dict(sim["material"])["density"]) ≈ 7800.0
moduli = as_array(as_dict(sim["material"])["moduli"])
@test as_float(moduli[1]) ≈ 210e9
bcs = as_array(sim["boundary_conditions"])
@test length(bcs) == 2
@test as_string(as_dict(bcs[1])["type"]) == "fixed"
@test as_float(as_dict(bcs[2])["value"]) ≈ -1000.0

# ── quoted keys ───────────────────────────────────────────────────────────────
println("=== Quoted keys ===")
v = SimpleYAML.loads("""
"key with spaces": 1
'another key': 2
""")
d = as_dict(v)
@test as_int(d["key with spaces"]) == 1
@test as_int(d["another key"]) == 2

# ── document separator ────────────────────────────────────────────────────────
println("=== Document separator ===")
v = SimpleYAML.loads("""
---
name: test
age: 1
""")
d = as_dict(v)
@test as_string(d["name"]) == "test"

# ── FEM-style input file ───────────────────────────────────────────────────────
println("=== FEM-style input ===")
fem_yaml = """
# Finite Element Simulation Input
---
problem:
  name: "Cantilever Beam"
  type: static_structural
  dimensions: 3

mesh:
  file: mesh/beam.msh
  order: 2
  refinement_level: 0

material:
  name: Steel
  density: 7850.0         # kg/m^3
  youngs_modulus: 2.1e11  # Pa
  poissons_ratio: 0.3

boundary_conditions:
  - name: fixed_end
    type: dirichlet
    dof: [u, v, w]
    value: 0.0
    nodesets: [1]

  - name: tip_load
    type: neumann
    component: w
    value: -5000.0
    facesets: [2]

solver:
  type: direct
  library: MUMPS
  tolerance: 1.0e-10
  max_iterations: 100

output:
  format: vtk
  fields:
    - displacement
    - stress
    - strain
  frequency: 1
"""

v = SimpleYAML.loads(fem_yaml)
d = as_dict(v)
display(d)
prob = as_dict(d["problem"])
@test as_string(prob["name"]) == "Cantilever Beam"
@test as_string(prob["type"]) == "static_structural"
@test as_int(prob["dimensions"]) == 3

mat = as_dict(d["material"])
@test as_string(mat["name"]) == "Steel"
@test as_float(mat["density"]) ≈ 7850.0
@test as_float(mat["youngs_modulus"]) ≈ 2.1e11
@test as_float(mat["poissons_ratio"]) ≈ 0.3

bcs = as_array(d["boundary_conditions"])
@test length(bcs) == 2
bc1 = as_dict(bcs[1])
@test as_string(bc1["name"]) == "fixed_end"
@test as_string(bc1["type"]) == "dirichlet"
dof = as_array(bc1["dof"])
@test as_string(dof[1]) == "u"
@test as_string(dof[3]) == "w"

bc2 = as_dict(bcs[2])
@test as_float(bc2["value"]) ≈ -5000.0

solver = as_dict(d["solver"])
@test as_string(solver["type"]) == "direct"
@test as_float(solver["tolerance"]) ≈ 1.0e-10

output = as_dict(d["output"])
fields = as_array(output["fields"])
@test length(fields) == 3
@test as_string(fields[2]) == "stress"

# ── summary ───────────────────────────────────────────────────────────────────
println()
println("Results: $pass passed, $fail failed out of $(pass+fail) tests")
fail > 0 && exit(1)
