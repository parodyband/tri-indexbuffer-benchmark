package benchmark

import "core:fmt"
import "core:time"
import "core:os"
import "core:math"
import "core:math/linalg"
import "core:sys/windows"
import "core:strings"
import rt "base:runtime"

import gl "vendor:OpenGL"
import "vendor:glfw"

import assimp ".."

GL_VERTEX_SHADER        :: gl.VERTEX_SHADER
GL_FRAGMENT_SHADER      :: gl.FRAGMENT_SHADER
GL_COMPILE_STATUS       :: gl.COMPILE_STATUS
GL_LINK_STATUS          :: gl.LINK_STATUS
GL_ARRAY_BUFFER         :: gl.ARRAY_BUFFER
GL_ELEMENT_ARRAY_BUFFER :: gl.ELEMENT_ARRAY_BUFFER
GL_STATIC_DRAW          :: gl.STATIC_DRAW
GL_FLOAT                :: gl.FLOAT
GL_UNSIGNED_INT         :: gl.UNSIGNED_INT
GL_FALSE                :: gl.FALSE
GL_TRUE                 :: gl.TRUE
GL_TRIANGLES            :: gl.TRIANGLES
GL_COLOR_BUFFER_BIT     :: gl.COLOR_BUFFER_BIT
GL_DEPTH_BUFFER_BIT     :: gl.DEPTH_BUFFER_BIT
GL_DEPTH_TEST           :: gl.DEPTH_TEST
GL_DRAW_INDIRECT_BUFFER :: gl.DRAW_INDIRECT_BUFFER

// Camera and transform utilities
Camera :: struct {
    position: [3]f32,
    target:   [3]f32,
    up:       [3]f32,
    fov:      f32,
    aspect:   f32,
    near:     f32,
    far:      f32,
}

// Calculate camera position based on grid size
calculate_camera_position :: proc(instance_count: int) -> [3]f32 {
    grid_size := int(math.sqrt(f32(instance_count)))
    if grid_size * grid_size < instance_count do grid_size += 1
    
    spacing := f32(4.0)
    grid_world_size := f32(grid_size) * spacing
    
    fov_radians := math.to_radians_f32(60)
    min_distance := (grid_world_size * 0.5) / math.tan(fov_radians * 0.5)
    
    camera_distance := min_distance * 1.2
    camera_height := camera_distance * 0.5
    
    return [3]f32{0, camera_height, camera_distance}
}

create_camera :: proc(instance_count: int) -> Camera {
    camera_pos := calculate_camera_position(instance_count)
    return Camera{
        position = camera_pos,
        target   = {0, 0, 0},     // Looking at origin
        up       = {0, 1, 0},     // Y-up
        fov      = 60,            // 60 degree FOV
        aspect   = 1.0,           // Will be set based on window
        near     = 0.1,
        far      = 10000.0,       // Increased far plane for larger scenes
    }
}

get_view_matrix :: proc(camera: ^Camera) -> matrix[4, 4]f32 {
    return linalg.matrix4_look_at_f32(camera.position, camera.target, camera.up)
}

get_projection_matrix :: proc(camera: ^Camera) -> matrix[4, 4]f32 {
    return linalg.matrix4_perspective_f32(math.to_radians_f32(camera.fov), 
                                        camera.aspect,
                                        camera.near, 
                                        camera.far)
}

// grid positions for instances
calculate_grid_positions :: proc(instance_count: int) -> []f32 {
    grid_size := int(math.sqrt(f32(instance_count)))
    if grid_size * grid_size < instance_count do grid_size += 1
    
    spacing := f32(4.0)
    offset_x := -f32(grid_size) * spacing * 0.5
    offset_z := -f32(grid_size) * spacing * 0.5
    
    positions := make([]f32, instance_count * 16)
    
    for i := 0; i < instance_count; i += 1 {
        row := i / grid_size
        col := i % grid_size
        
        x := offset_x + f32(col) * spacing
        z := offset_z + f32(row) * spacing
        
        model := linalg.MATRIX4F32_IDENTITY
        translate_vec := linalg.Vector3f32{x, 0, z}
        scale_vec := linalg.Vector3f32{1.0, 1.0, 1.0}
        model = linalg.matrix4_translate_f32(translate_vec) * model
        model = linalg.matrix4_scale_f32(scale_vec) * model
        
        // Copy matrix to buffer
        offset := i * 16
        for r := 0; r < 4; r += 1 {
            for c := 0; c < 4; c += 1 {
                positions[offset + r*4 + c] = model[r][c]
            }
        }
    }
    
    return positions
}

MeshData :: struct {
    vertices     : []f32,
    uvs          : []f32,
    indices      : []u32,
    vertex_count : int,
    index_count  : int,

    vao : u32,
    vbo : u32,
    uv_buffer: u32,
    ebo : u32,
    instance_buffer: u32,
}

// Assimp
load_mesh_data :: proc(file_path: string) -> MeshData {
    c_path := strings.clone_to_cstring(file_path);
    defer delete(c_path);
    
    flags := u32(assimp.aiPostProcessSteps.Triangulate) | 
             u32(assimp.aiPostProcessSteps.JoinIdenticalVertices) | 
             u32(assimp.aiPostProcessSteps.ImproveCacheLocality);
    
    scene := assimp.import_file(c_path, flags);
    defer assimp.release_import(scene);

    if scene == nil {
        fmt.println("ERROR: Could not load mesh from:", file_path);
        return MeshData{};
    }

    if scene.mNumMeshes == 0 {
        fmt.println("ERROR: No meshes in file:", file_path);
        return MeshData{};
    }

    meshes := ([^]^assimp.aiMesh)(scene.mMeshes);
    mesh_ptr := meshes[0];
    if mesh_ptr == nil {
        fmt.println("ERROR: First mesh is null in:", file_path);
        return MeshData{};
    }
    mesh := mesh_ptr^;

    // Positions
    vertex_count := int(mesh.mNumVertices);
    positions := make([]f32, vertex_count * 3);
    vertices := ([^]assimp.aiVector3D)(mesh.mVertices);
    for i in 0..<vertex_count {
        positions[i*3+0] = vertices[i].x;
        positions[i*3+1] = vertices[i].y;
        positions[i*3+2] = vertices[i].z;
    }

    // UVs (texture coordinates)
    uvs := make([]f32, vertex_count * 2);
    if mesh.mTextureCoords[0] != nil {
        tex_coords := ([^]assimp.aiVector3D)(mesh.mTextureCoords[0]);
        for i in 0..<vertex_count {
            uvs[i*2+0] = tex_coords[i].x;
            uvs[i*2+1] = tex_coords[i].y;
        }
    } else {
        for i in 0..<vertex_count {
            uvs[i*2+0] = positions[i*3+0];  // Use X coordinate
            uvs[i*2+1] = positions[i*3+2];  // Use Z coordinate
        }
    }

    // Indices
    total_index_count: u32 = 0;
    faces := ([^]assimp.aiFace)(mesh.mFaces);
    for f in 0..<mesh.mNumFaces {
        total_index_count += faces[f].mNumIndices;
    }
    indices := make([]u32, total_index_count);

    offset: u32 = 0;
    for f in 0..<mesh.mNumFaces {
        face := faces[f];
        face_indices := ([^]u32)(face.mIndices);
        for idx in 0..<face.mNumIndices {
            indices[offset] = face_indices[idx];
            offset += 1;
        }
    }

    return MeshData{
        vertices = positions,
        uvs = uvs,
        indices = indices,
        vertex_count = vertex_count,
        index_count = int(total_index_count),
    };
}

create_vao_for_mesh :: proc(mesh: ^MeshData, instance_count: int) {
    // Generate buffers
    gl.GenVertexArrays(1, &mesh.vao);
    gl.GenBuffers(1, &mesh.vbo);
    gl.GenBuffers(1, &mesh.uv_buffer);
    gl.GenBuffers(1, &mesh.ebo);
    gl.GenBuffers(1, &mesh.instance_buffer);

    gl.BindVertexArray(mesh.vao);

    // Vertex buffer
    gl.BindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    gl.BufferData(GL_ARRAY_BUFFER,
                  len(mesh.vertices) * size_of(f32),
                  raw_data(mesh.vertices),
                  GL_STATIC_DRAW);

    // UV buffer
    gl.BindBuffer(GL_ARRAY_BUFFER, mesh.uv_buffer);
    gl.BufferData(GL_ARRAY_BUFFER,
                  len(mesh.uvs) * size_of(f32),
                  raw_data(mesh.uvs),
                  GL_STATIC_DRAW);

    // Index buffer
    gl.BindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);
    gl.BufferData(GL_ELEMENT_ARRAY_BUFFER,
                  len(mesh.indices) * size_of(u32),
                  raw_data(mesh.indices),
                  GL_STATIC_DRAW);

    // Instance buffer
    instance_data := calculate_grid_positions(instance_count);
    defer delete(instance_data);
    
    gl.BindBuffer(GL_ARRAY_BUFFER, mesh.instance_buffer);
    gl.BufferData(GL_ARRAY_BUFFER,
                  len(instance_data) * size_of(f32),
                  raw_data(instance_data),
                  GL_STATIC_DRAW);

    // Vertex attribute: position (layout = 0)
    gl.BindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    gl.EnableVertexAttribArray(0);
    gl.VertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*size_of(f32), uintptr(0));

    // Vertex attribute: UV (layout = 1)
    gl.BindBuffer(GL_ARRAY_BUFFER, mesh.uv_buffer);
    gl.EnableVertexAttribArray(1);
    gl.VertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2*size_of(f32), uintptr(0));

    // Instance matrix attributes (layout = 2-5)
    gl.BindBuffer(GL_ARRAY_BUFFER, mesh.instance_buffer);
    mat_size := 4 * size_of(f32)
    for i in 0..<4 {
        gl.EnableVertexAttribArray(u32(2 + i));
        gl.VertexAttribPointer(u32(2 + i), 4, GL_FLOAT, GL_FALSE, 16*size_of(f32), uintptr(i * mat_size));
        gl.VertexAttribDivisor(u32(2 + i), 1);  // This is an instance attribute
    }

    // Unbind VAO
    gl.BindVertexArray(0);
}

compile_shader :: proc(src: string, shader_type: u32) -> u32 {
    shader := gl.CreateShader(shader_type);

    // Convert Odin string -> null-terminated cstring
    src_cstr := strings.clone_to_cstring(src);
    defer delete(src_cstr);
    
    src_ptr := (^cstring)(&src_cstr);
    gl.ShaderSource(shader, 1, src_ptr, nil);
    gl.CompileShader(shader);

    success: i32;
    gl.GetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if success == 0 {
        info_log := make([]u8, 512);
        gl.GetShaderInfoLog(shader, 512, nil, raw_data(info_log));
        fmt.println("Shader compile error:", string(info_log));
    }
    return shader;
}

// Create shader program
create_shader_program :: proc() -> u32 {
    vertex_src := strings.concatenate({
        "#version 330 core\n",
        "layout (location = 0) in vec3 aPos;\n",
        "layout (location = 1) in vec2 aTexCoord;\n",
        // Instance matrix (4 vec4s)
        "layout (location = 2) in mat4 instanceMatrix;\n",
        "uniform mat4 view;\n",
        "uniform mat4 projection;\n",
        "out vec2 TexCoord;\n",
        "void main() {\n",
        "    gl_Position = projection * view * instanceMatrix * vec4(aPos, 1.0);\n",
        "    TexCoord = aTexCoord * 8.0;\n", 
        "}\n",
    });
    defer delete(vertex_src);

    fragment_src := strings.concatenate({
        "#version 330 core\n",
        "in vec2 TexCoord;\n",
        "out vec4 FragColor;\n",
        "void main() {\n",
        "    vec2 pattern = floor(TexCoord);\n",
        "    float checker = mod(pattern.x + pattern.y, 2.0);\n",
        "    vec3 color1 = vec3(0.2, 0.3, 0.8);\n",  // Blue-ish
        "    vec3 color2 = vec3(0.8, 0.3, 0.2);\n",  // Orange-ish
        "    vec3 color = mix(color1, color2, checker);\n",
        "    FragColor = vec4(color, 1.0);\n",
        "}\n",
    });
    defer delete(fragment_src);

    vs := compile_shader(vertex_src, GL_VERTEX_SHADER);
    fs := compile_shader(fragment_src, GL_FRAGMENT_SHADER);

    program := gl.CreateProgram();
    gl.AttachShader(program, vs);
    gl.AttachShader(program, fs);
    gl.LinkProgram(program);

    success: i32;
    gl.GetProgramiv(program, GL_LINK_STATUS, &success);
    if success == 0 {
        info_log := make([]u8, 512);
        gl.GetProgramInfoLog(program, 512, nil, raw_data(info_log));
        fmt.println("Program link error:", string(info_log));
    }

    gl.DeleteShader(vs);
    gl.DeleteShader(fs);

    return program;
}

run_benchmark :: proc(mesh: MeshData, program: u32, instance_count: int, warmup_frames: int, measure_frames: int, camera: ^Camera) -> f64 {
    fmt.printf("Running benchmark with %d instances...\n", instance_count);
    
    gl.Enable(GL_DEPTH_TEST);
    
    view_loc := gl.GetUniformLocation(program, "view");
    proj_loc := gl.GetUniformLocation(program, "projection");
    
    view := get_view_matrix(camera);
    proj := get_projection_matrix(camera);
    
    view_data := transmute([16]f32)view;
    proj_data := transmute([16]f32)proj;


    time.sleep(50 * time.Millisecond);

    // Multi-draw indirect commands
    batches := 1;
    if instance_count >= 100_000 {
        batches = 4;  // Fewer, larger batches
    }
    batch_size := instance_count / batches;
    remainder := instance_count % batches;
    total_offset := 0;
    commands := make([]u32, batches * 5);  // 5 values per command: count, instanceCount, firstIndex, baseVertex, baseInstance
    for i := 0; i < batches; i += 1 {
        inst_count := batch_size;
        if i < remainder {
            inst_count += 1;
        }
        commands[i*5 + 0] = u32(mesh.index_count);
        commands[i*5 + 1] = u32(inst_count);
        commands[i*5 + 2] = 0;
        commands[i*5 + 3] = 0;
        commands[i*5 + 4] = u32(total_offset);
        total_offset += inst_count;
    }
    indirect_buffer: u32;
    gl.GenBuffers(1, &indirect_buffer);
    gl.BindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_buffer);
    gl.BufferData(GL_DRAW_INDIRECT_BUFFER,
                  len(commands) * size_of(u32),
                  raw_data(commands),
                  GL_STATIC_DRAW);

    // Warm-up frames (not timed)
    for _ in 0..<warmup_frames {
        gl.ClearColor(0.1, 0.1, 0.1, 1.0)  // Dark gray background
        gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        gl.UseProgram(program)

        // Set uniforms
        gl.UniformMatrix4fv(view_loc, 1, GL_FALSE, raw_data(&view_data))
        gl.UniformMatrix4fv(proj_loc, 1, GL_FALSE, raw_data(&proj_data))

        gl.BindVertexArray(mesh.vao)
        gl.BindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_buffer)
        gl.MultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, nil, i32(batches), 5 * size_of(u32))

        glfw.SwapBuffers(glfw.GetCurrentContext())
        glfw.PollEvents()
    }

    // Timed frames
    start_time := time.now()
    for _ in 0..<measure_frames {
        gl.ClearColor(0.1, 0.1, 0.1, 1.0)
        gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        gl.UseProgram(program)

        // Set uniforms
        gl.UniformMatrix4fv(view_loc, 1, GL_FALSE, raw_data(&view_data))
        gl.UniformMatrix4fv(proj_loc, 1, GL_FALSE, raw_data(&proj_data))

        gl.BindVertexArray(mesh.vao)
        gl.BindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_buffer)
        gl.MultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, nil, i32(batches), 5 * size_of(u32))

        glfw.SwapBuffers(glfw.GetCurrentContext())
        glfw.PollEvents()
    }
    gl.DeleteBuffers(1, &indirect_buffer)

    end_time := time.now()
    duration := time.diff(start_time, end_time)
    total_ms := f64(duration) / 1e6
    avg_ms_per_frame := total_ms / f64(measure_frames)
    return avg_ms_per_frame
}

write_benchmark_results :: proc(file_path: string, test_number: int, mesh_variant: string, instance_count: int, avg_ms: f64) {
    file, err := os.open(file_path, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o644);
    if err != 0 {
        fmt.printf("Error opening file %s: %v\n", file_path, err);
        return;
    }
    defer os.close(file);

    fmt.fprintf(file, "%d,%s,%d,%.3f\n", test_number, mesh_variant, instance_count, avg_ms);
}

main :: proc() {
    if !glfw.Init() {
        fmt.println("Failed to init GLFW");
        return;
    }
    defer glfw.Terminate();

    glfw.WindowHint(glfw.CLIENT_API, glfw.OPENGL_API);
    glfw.WindowHint(glfw.CONTEXT_VERSION_MAJOR, 4);
    glfw.WindowHint(glfw.CONTEXT_VERSION_MINOR, 3);
    glfw.WindowHint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE); 
    glfw.WindowHint(glfw.RESIZABLE, 0); 

    window := glfw.CreateWindow(2560, 1440, "Bullet Benchmarker", nil, nil);
    if window == nil {
        fmt.println("Failed to create GLFW window");
        return;
    }
    glfw.MakeContextCurrent(window);

    gl.load_up_to(4, 3, glfw.gl_set_proc_address);

    meshA_path := "bullet_chamfer.obj";
    meshB_path := "bullet_uv.obj";

    meshA := load_mesh_data(meshA_path);
    meshB := load_mesh_data(meshB_path);

    program := create_shader_program();

    // Instance counts for your benchmark (exponential growth)
    instance_counts := [15]int{
        100,     // 0.1K
        500,     // 0.5K
        1000,    // 1K
        5000,    // 5K
        10000,   // 10K
        25000,   // 25K
        50000,   // 50K
        75000,   // 75K
        100000,  // 100K
        150000,  // 150K
        200000,  // 200K
        250000,  // 250K
        300000,  // 300K
        400000,  // 400K
        500000,  // 500K
    };

    warmup_frames := 20;
    measure_frames := 100;

    results_file := "benchmark_results.csv";
    {
        file, err := os.open(results_file, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644);
        if err != 0 {
            fmt.printf("Error creating file %s: %v\n", results_file, err);
            return;
        }
        fmt.fprintf(file, "test_number,mesh_variant,instance_count,avg_ms_per_frame\n");
        os.close(file);
    }

    fmt.println("test_number,mesh_variant,instance_count,avg_ms_per_frame");

    // Run the benchmark 3 times
    for test_number in 1..=3 {
        fmt.printf("\nStarting Test Run %d\n", test_number);
        
        for ic in instance_counts {
            camera := create_camera(ic);
            camera.aspect = 2560.0/1440.0;
            
            create_vao_for_mesh(&meshA, ic);
            avg_ms := run_benchmark(meshA, program, ic, warmup_frames, measure_frames, &camera);
            fmt.printf("%d,A,%d,%.3f\n", test_number, ic, avg_ms);
            write_benchmark_results(results_file, test_number, "A", ic, avg_ms);
            
            gl.Finish();
            time.sleep(100 * time.Millisecond);
        }

        for ic in instance_counts {
            camera := create_camera(ic);
            camera.aspect = 2560.0/1440.0;
            
            create_vao_for_mesh(&meshB, ic);
            avg_ms := run_benchmark(meshB, program, ic, warmup_frames, measure_frames, &camera);
            fmt.printf("%d,B,%d,%.3f\n", test_number, ic, avg_ms);
            write_benchmark_results(results_file, test_number, "B", ic, avg_ms);
            
            gl.Finish();
            time.sleep(100 * time.Millisecond);
        }

        if test_number < 3 {
            fmt.printf("\nWaiting 2 seconds before next test run...\n");
            time.sleep(2 * time.Second);
        }
    }

    fmt.printf("\nBenchmark results have been saved to: %s\n", results_file);

    glfw.SetWindowShouldClose(window, true);
}
