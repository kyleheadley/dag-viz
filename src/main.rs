#[macro_use]
extern crate glium;
extern crate glium_text;
extern crate cgmath;
extern crate image;
extern crate time;
extern crate rand;
mod util;

use std::default::Default;
use std::io::{Cursor, Read};
use std::fs::File;
use std::env::current_exe;
use std::collections::HashMap;
use glium::{DisplayBuild, Surface, VertexBuffer, IndexBuffer};
use glium::glutin::{WindowBuilder, Event, VirtualKeyCode, ElementState, MouseButton};
use cgmath::{Quaternion, Decomposed, Matrix4, Vector3, Point3, Deg};
use cgmath::{Transform, Angle, Rotation, Rotation3, Zero, One};
use cgmath::{InnerSpace, EuclideanSpace, Matrix};
use rand::Rng;

// preferences
const ZOOM_SCALE:f32 = 1.1; // multiplier
const MOVE_SPEED:f32 = 2.0;
const TURN_SPEED:f32 = 5.0;
const LIGHT_ROTATE_SPEED:f32 = 0.1;
const TEXT_LINE_HEIGHT:f32 = 50.0;
const LIGHT_SIZE:f32 = 0.2;
const AMBIENT_LIGHT:[f32; 3] = [0.04,0.04,0.04];
const HL_NODE_SCALE: f32 = 1.1;
const HL_EDGE_SCALE: f32 = 1.5;

fn main() {
  // show window
  let display =
    WindowBuilder::new()
    .with_depth_buffer(24)
    .with_title("DAGViz")
    .with_vsync()
    .build_glium()
    .unwrap();

  // Our visual models
  let mut vertex_models = HashMap::new();

  // define cube model
  let mut cube_raw_verts:Vec<Vertex> = Vec::new();
  let mut cube_raw_indices:Vec<u16> = Vec::new();
  //front
  cube_raw_verts.push(Vertex{
    position:(-1.0,1.0,-1.0), normal: (0.0,0.0,-1.0), tex_coords: (0.0,0.0),
  });
  cube_raw_verts.push(Vertex{
    position:(1.0,1.0,-1.0), normal: (0.0,0.0,-1.0), tex_coords: (0.0,0.0),
  });
  cube_raw_verts.push(Vertex{
    position:(1.0,-1.0,-1.0), normal: (0.0,0.0,-1.0), tex_coords: (0.0,0.0),
  });
  cube_raw_verts.push(Vertex{
    position:(-1.0,-1.0,-1.0), normal: (0.0,0.0,-1.0), tex_coords: (0.0,0.0),
  });
  //back
  cube_raw_verts.push(Vertex{
    position:(1.0,1.0,1.0), normal: (0.0,0.0,1.0), tex_coords: (0.0,0.0),
  });
  cube_raw_verts.push(Vertex{
    position:(-1.0,1.0,1.0), normal: (0.0,0.0,1.0), tex_coords: (0.0,0.0),
  });
  cube_raw_verts.push(Vertex{
    position:(-1.0,-1.0,1.0), normal: (0.0,0.0,1.0), tex_coords: (0.0,0.0),
  });
  cube_raw_verts.push(Vertex{
    position:(1.0,-1.0,1.0), normal: (0.0,0.0,1.0), tex_coords: (0.0,0.0),
  });
  //right
  cube_raw_verts.push(Vertex{
    position:(1.0,1.0,-1.0), normal: (1.0,0.0,0.0), tex_coords: (0.0,0.0),
  });
  cube_raw_verts.push(Vertex{
    position:(1.0,1.0,1.0), normal: (1.0,0.0,0.0), tex_coords: (0.0,0.0),
  });
  cube_raw_verts.push(Vertex{
    position:(1.0,-1.0,1.0), normal: (1.0,0.0,0.0), tex_coords: (0.0,0.0),
  });
  cube_raw_verts.push(Vertex{
    position:(1.0,-1.0,-1.0), normal: (1.0,0.0,0.0), tex_coords: (0.0,0.0),
  });
  //left
  cube_raw_verts.push(Vertex{
    position:(-1.0,1.0,1.0), normal: (-1.0,0.0,0.0), tex_coords: (0.0,0.0),
  });
  cube_raw_verts.push(Vertex{
    position:(-1.0,1.0,-1.0), normal: (-1.0,0.0,0.0), tex_coords: (0.0,0.0),
  });
  cube_raw_verts.push(Vertex{
    position:(-1.0,-1.0,-1.0), normal: (-1.0,0.0,0.0), tex_coords: (0.0,0.0),
  });
  cube_raw_verts.push(Vertex{
    position:(-1.0,-1.0,1.0), normal: (-1.0,0.0,0.0), tex_coords: (0.0,0.0),
  });
  //top
  cube_raw_verts.push(Vertex{
    position:(1.0,1.0,-1.0), normal: (0.0,1.0,0.0), tex_coords: (0.0,0.0),
  });
  cube_raw_verts.push(Vertex{
    position:(-1.0,1.0,-1.0), normal: (0.0,1.0,0.0), tex_coords: (0.0,0.0),
  });
  cube_raw_verts.push(Vertex{
    position:(-1.0,1.0,1.0), normal: (0.0,1.0,0.0), tex_coords: (0.0,0.0),
  });
  cube_raw_verts.push(Vertex{
    position:(1.0,1.0,1.0), normal: (0.0,1.0,0.0), tex_coords: (0.0,0.0),
  });
  //bottom
  cube_raw_verts.push(Vertex{
    position:(-1.0,-1.0,-1.0), normal: (0.0,-1.0,0.0), tex_coords: (0.0,0.0),
  });
  cube_raw_verts.push(Vertex{
    position:(1.0,-1.0,-1.0), normal: (0.0,-1.0,0.0), tex_coords: (0.0,0.0),
  });
  cube_raw_verts.push(Vertex{
    position:(1.0,-1.0,1.0), normal: (0.0,-1.0,0.0), tex_coords: (0.0,0.0),
  });
  cube_raw_verts.push(Vertex{
    position:(-1.0,-1.0,1.0), normal: (0.0,-1.0,0.0), tex_coords: (0.0,0.0),
  });
  for i in 0..6 {
    let i = i * 4;
    cube_raw_indices.append(&mut vec!(i + 0,i + 1,i + 2));
    cube_raw_indices.append(&mut vec!(i + 0,i + 2,i + 3));
  }
  let cube_verts = VertexBuffer::new(&display, &cube_raw_verts).unwrap();
  let cube_indices =
    IndexBuffer::new(
      &display,
      glium::index::PrimitiveType::TrianglesList,
      &cube_raw_indices)
    .unwrap();
  vertex_models.insert("Cube", (cube_verts, cube_indices));
  // sphere model
  let mut sphere_raw_verts:Vec<Vertex> = Vec::new();
  let mut sphere_raw_indices:Vec<u16> = Vec::new();
  let w:u16 = 10;
  let h:u16 = 10;
  for i in 0..w {
    let long0 = cgmath::Deg(i as f32 * 360.0 / (w as f32));
    let long1 = cgmath::Deg((i+1) as f32 * 360.0 / (w as f32));
    for j in 0..h {
      let lat0 = cgmath::Deg(j as f32 * 180.0 / (h as f32));
      let lat1 = cgmath::Deg((j+1) as f32 * 180.0 / (h as f32));
      sphere_raw_verts.push(
        Vertex{
          position:(long0.sin()*lat0.sin(),lat0.cos(),long0.cos()*lat0.sin()),
          normal: (long0.sin()*lat0.sin(),lat0.cos(),long0.cos()*lat0.sin()),
          tex_coords: (0.0,0.0),
        });
      sphere_raw_verts.push(
        Vertex{
          position:(long0.sin()*lat1.sin(),lat1.cos(),long0.cos()*lat1.sin()),
          normal: (long0.sin()*lat1.sin(),lat1.cos(),long0.cos()*lat1.sin()),
          tex_coords: (0.0,0.0),
        });
      sphere_raw_verts.push(
        Vertex{
          position:(long1.sin()*lat1.sin(),lat1.cos(),long1.cos()*lat1.sin()),
          normal: (long1.sin()*lat1.sin(),lat1.cos(),long1.cos()*lat1.sin()),
          tex_coords: (0.0,0.0),
        });
      sphere_raw_verts.push(
        Vertex{
          position:(long1.sin()*lat0.sin(),lat0.cos(),long1.cos()*lat0.sin()),
          normal: (long1.sin()*lat0.sin(),lat0.cos(),long1.cos()*lat0.sin()),
          tex_coords: (0.0,0.0),
        });
      let count:u16 = i * h * 4 + j * 4;
      sphere_raw_indices.append(&mut vec!(count + 0,count + 1,count + 2));
      sphere_raw_indices.append(&mut vec!(count + 0,count + 2,count + 3));
    }
  }
  let sphere_verts = VertexBuffer::new(&display, &sphere_raw_verts).unwrap();
  let sphere_indices =
    IndexBuffer::new(
      &display,
      glium::index::PrimitiveType::TrianglesList,
      &sphere_raw_indices)
    .unwrap();
  vertex_models.insert("Sphere", (sphere_verts, sphere_indices));
  // cylinder model for edges
  // - aligned on z-axis
  // - TODO: add end caps?
  let w:u16 = 10;
  let h:u16 = 10;
  let cyl_width = 0.1;
  let mut cyl_raw_verts:Vec<Vertex> = Vec::new();
  let mut cyl_raw_indices:Vec<u16> = Vec::new();
  for i in 0..w {
    let long0 = cgmath::Deg(i as f32 * 360.0 / (w as f32));
    let long1 = cgmath::Deg((i+1) as f32 * 360.0 / (w as f32));
    for j in 0..h {
      let lat0 = j as f32 / (h as f32);
      let lat1 = (j+1) as f32 / (h as f32);
      cyl_raw_verts.push(
        Vertex{
          position:(long0.cos() * cyl_width,long0.sin() * cyl_width,-lat0),
          normal: (long0.cos(),long0.sin(),0.0),
          tex_coords: (0.0,0.0),
        });
      cyl_raw_verts.push(
        Vertex{
          position:(long0.cos() * cyl_width,long0.sin() * cyl_width,-lat1),
          normal: (long0.cos(),long0.sin(),0.0),
          tex_coords: (0.0,0.0),
        });
      cyl_raw_verts.push(
        Vertex{
          position:(long1.cos() * cyl_width,long1.sin() * cyl_width,-lat1),
          normal: (long1.cos(),long1.sin(),0.0),
          tex_coords: (0.0,0.0),
        });
      cyl_raw_verts.push(
        Vertex{
          position:(long1.cos() * cyl_width,long1.sin() * cyl_width,-lat0),
          normal: (long1.cos(),long1.sin(),0.0),
          tex_coords: (0.0,0.0),
        });
      let count:u16 = i * h * 4 + j * 4;
      cyl_raw_indices.append(&mut vec!(count + 0,count + 1,count + 2));
      cyl_raw_indices.append(&mut vec!(count + 0,count + 2,count + 3));
    }
  }
  let cyl_verts = VertexBuffer::new(&display, &cyl_raw_verts).unwrap();
  let cyl_indices =
    IndexBuffer::new(
      &display,
      glium::index::PrimitiveType::TrianglesList,
      &cyl_raw_indices)
    .unwrap();
  vertex_models.insert("Cylinder", (cyl_verts, cyl_indices));
  // pyramid model for edge arrow caps
  // - TODO: create model

  // load files
  let src_path = current_exe().unwrap().parent().unwrap().join(std::path::Path::new("../../src/"));
  let mut model_v_shader = String::new();
  File::open(&src_path.join("mv_shader.glsl")).unwrap().read_to_string(&mut model_v_shader).unwrap();
  let mut model_f_shader = String::new();
  File::open(&src_path.join("mf_shader.glsl")).unwrap().read_to_string(&mut model_f_shader).unwrap();
  let mut picker_v_shader = String::new();
  File::open(&src_path.join("pv_shader.glsl")).unwrap().read_to_string(&mut picker_v_shader).unwrap();
  let mut picker_f_shader = String::new();
  File::open(&src_path.join("pf_shader.glsl")).unwrap().read_to_string(&mut picker_f_shader).unwrap();
  let mut highlight_v_shader = String::new();
  File::open(&src_path.join("hv_shader.glsl")).unwrap().read_to_string(&mut highlight_v_shader).unwrap();
  let mut highlight_f_shader = String::new();
  File::open(&src_path.join("hf_shader.glsl")).unwrap().read_to_string(&mut highlight_f_shader).unwrap();
  let font = glium_text::FontTexture::new(&display, std::fs::File::open(&src_path.join("FiraMono-Bold.ttf")).unwrap(), 24).unwrap();
  let small_texture = util::tex2d_load(&display, Cursor::new(&include_bytes!("Small.png")[..]),image::PNG);
  let text_system = glium_text::TextSystem::new(&display);
  let picker_output = glium::texture::Texture2d::empty(&display,1,1).unwrap();

  // set up default data
  let mut camera = Camera::default();
  let mut light = Point3::new(-10.0, 4.0, 9.0);
  let mut rotate_light = true;
  let mut graph = DAGraph::new();
  let mut selected = 0;

  // build random graph
  let nodes = 50;
  let to_node = 1; // to two more
  let from_range = 10;
  let graph_width = 30.0;
  let mut rng = rand::thread_rng();
  for i in 0..nodes {
    graph.add_node(
      Decomposed{
        scale: rng.gen::<f32>() + 0.5,
        rot: Quaternion::one(),
        disp: cgmath::vec3(
          rng.gen::<f32>()*graph_width-graph_width/2.0,
          graph_width-2.0*graph_width/nodes as f32*i as f32,
          rng.gen::<f32>()*graph_width-graph_width/2.0,
        ),
      }.into(),
      Material{
        diffuse_color: [rng.gen::<f32>(),rng.gen::<f32>(),rng.gen::<f32>()],
        specular_color: [0.8,0.8,0.8],
        model: if rng.gen() {"Sphere".to_string()} else {"Cube".to_string()},
      },
      vec!(format!("Node: {:?}", i))
    );
    if i > 0 {
      // make a few connections to this node from previous ones
      for _ in 0..(rng.gen::<usize>()%3 + to_node) {
        let from_node = if i < from_range {
          rng.gen::<usize>() % i
        } else {
          rng.gen::<usize>() % from_range + i - from_range
        };
        graph.add_edge(
          from_node,
          i,
          Material{
            diffuse_color: [rng.gen::<f32>(),rng.gen::<f32>(),rng.gen::<f32>()],
            specular_color: [0.8,0.8,0.8],
            model: "None".to_string(),
          },
          vec!(format!("Edge: {:?}", i))
        );
      }
    }
  }

// set up shaders
  let model_program =
    glium::Program::from_source(
      &display,
      &model_v_shader, &model_f_shader,
      None)
    .unwrap();
  let picker_program =
    glium::Program::from_source(
      &display,
      &picker_v_shader, &picker_f_shader,
      None)
    .unwrap();
  let highlight_program =
    glium::Program::from_source(
      &display,
      &highlight_v_shader, &highlight_f_shader,
      None)
    .unwrap();

  // set up GL params (depth, culling)
  let params = glium::DrawParameters {
      depth: glium::Depth {
          test: glium::draw_parameters::DepthTest::IfLess,
          write: true,
          .. Default::default()
      },
      backface_culling: glium::draw_parameters::BackfaceCullingMode::CullCounterClockwise,
      .. Default::default()
  };
  let highlight_params = glium::DrawParameters {
      depth: glium::Depth {
          test: glium::draw_parameters::DepthTest::IfLess,
          write: true,
          .. Default::default()
      },
      blend: glium::Blend::alpha_blending(),
      backface_culling: glium::draw_parameters::BackfaceCullingMode::CullCounterClockwise,
      .. Default::default()
  };

  // timing
  let mut new_time = time::precise_time_ns();
  let mut old_time;
  let mut time_diff;

  // dev vars
  let mut _switch = false;
  let mut _dev_multiplier = 1.0;

  // ---------
  // Draw Loop
  // ---------
  loop {
    // set timing vars
    old_time = new_time;
    new_time = time::precise_time_ns();
    time_diff = (new_time - old_time) as f32 / 1_000_000.0;

    // create perspective
    let mut target = display.draw();
    let (width, height) = target.get_dimensions();
    let aspect_ratio = height as f32 / width as f32;
    let perspective = util::perspective(camera.fov, aspect_ratio, 1024.0);

    // initialize drawing surfaces
    target.clear_color_and_depth((0.0, 0.0, 0.1, 1.0), 1.0);
    let pic_tex = glium::texture::Texture2d::empty(&display, width, height).unwrap();
    let pic_dep = glium::texture::DepthTexture2d::empty(&display, width, height).unwrap();
    let mut picker_buf = glium::framebuffer::SimpleFrameBuffer::with_depth_buffer(&display,&pic_tex,&pic_dep).unwrap();
    picker_buf.clear_color_and_depth((0.0,0.0,0.0,0.0),1.0);

    // move light around origin
    if rotate_light {
      light = Quaternion::from_angle_y(Deg(LIGHT_ROTATE_SPEED * time_diff)).rotate_point(light);
    }

    // transform to view
    let inverse_cam = camera.transform.inverse_transform().unwrap();
    let transformed_light: [f32;3] = inverse_cam.transform_point(light).into();

    // draw light
    let modelview: Matrix4<f32> = inverse_cam * Into::<Matrix4<f32>>::into(Decomposed{
      scale: LIGHT_SIZE, rot: Quaternion::one(),
      disp: cgmath::vec3(light.x, light.y, light.z),
    });
    match vertex_models.get("Sphere") {
      None => {},
      Some(&(ref verts, ref idxs)) => {
        target.draw(
          verts, idxs,
          &model_program,
          &uniform! {
            modelview: Into::<[[f32;4];4]>::into(modelview),
            normal_mat: Into::<[[f32;4];4]>::into(modelview.inverse_transform().unwrap().transpose()), 
            ambient_l: [1.0,1.0,1.0f32], // emulate emmisive light
            diffuse_c: [1.0,1.0,1.0f32],
            specular_c: [0.0,0.0,0.0f32],
            perspective: perspective,
            u_light: transformed_light,
            tex: &small_texture, //using the white corner
          },
          &params
        ).unwrap();
      }
    }

    // draw graph
    for node in graph.nodes_iter() {
      let model: &str = &node.visual.model;
      match vertex_models.get(model) {
        None => {},
        Some(&(ref verts, ref idxs)) => {
          // premultiply
          let modelview = inverse_cam * node.transform;
          target.draw(
            verts, idxs,
            &model_program,
            &uniform! {
              modelview: Into::<[[f32;4];4]>::into(modelview),
              normal_mat: Into::<[[f32;4];4]>::into(modelview.inverse_transform().unwrap().transpose()),
              ambient_l: AMBIENT_LIGHT,
              diffuse_c: node.visual.diffuse_color,
              specular_c: node.visual.specular_color,
              perspective: perspective,
              u_light: transformed_light,
              tex: &small_texture, //using the white corner
            },
            &params
          ).unwrap();
          picker_buf.draw(
            verts, idxs,
            &picker_program,
            &uniform! {
              modelview: Into::<[[f32;4];4]>::into(modelview),
              normal_mat: Into::<[[f32;4];4]>::into(modelview.inverse_transform().unwrap().transpose()),
              perspective: perspective,
              u_id: util::uniform_of_id(node.id),
            },
            &params
          ).unwrap();
        }
      }
    }
    for edge in graph.edges_iter() {
      match vertex_models.get("Cylinder") {
        None => {},
        Some(&(ref verts, ref idxs)) => {
          // premultiply
          let modelview = inverse_cam * edge.transform;
          target.draw(
            verts, idxs,
            &model_program,
            &uniform! {
              modelview: Into::<[[f32;4];4]>::into(modelview),
              normal_mat: Into::<[[f32;4];4]>::into(modelview.inverse_transform().unwrap().transpose()),
              ambient_l: AMBIENT_LIGHT,
              diffuse_c: edge.visual.diffuse_color,
              specular_c: edge.visual.specular_color,
              perspective: perspective,
              u_light: transformed_light,
              tex: &small_texture, //using the white corner
            },
            &params
          ).unwrap();
          picker_buf.draw(
            verts, idxs,
            &picker_program,
            &uniform! {
              modelview: Into::<[[f32;4];4]>::into(modelview),
              normal_mat: Into::<[[f32;4];4]>::into(modelview.inverse_transform().unwrap().transpose()),
              perspective: perspective,
              u_id: util::uniform_of_id(edge.id),
            },
            &params
          ).unwrap();
        }
      }
    }
    // draw extra info on graph
    for node in graph.nodes_iter() {
      if node.is_highlight || node.is_source || node.is_sink {
        let model: &str = &node.visual.model;
        match vertex_models.get(model) {
          None => {},
          Some(&(ref verts, ref idxs)) => {
            let highlights: u32 = {
              let mut highlights = 0;
              if node.is_highlight {highlights |= 1}
              if node.is_source {highlights |= 2}
              if node.is_sink {highlights |= 4}
              highlights
            };
            // premultiply
            let modelview = inverse_cam * node.transform;
            let modelview = modelview * Matrix4::from_scale(HL_NODE_SCALE);
            target.draw(
              verts, idxs,
              &highlight_program,
              &uniform! {
                modelview: Into::<[[f32;4];4]>::into(modelview),
                perspective: perspective,
                direction: [0.0,-1.0,0.0f32],
                highlights: highlights,
                time: (new_time as f64 / 1_000_000_000.0) as f32,
              },
              &highlight_params
            ).unwrap();
          }
        }
      }
    }
    for edge in graph.edges_iter() {
      if edge.is_highlight || edge.is_source || edge.is_sink {
        match vertex_models.get("Cylinder") {
          None => {},
          Some(&(ref verts, ref idxs)) => {
            let highlights: u32 = {
              let mut highlights = 0;
              if edge.is_highlight {highlights |= 1}
              if edge.is_source {highlights |= 2}
              if edge.is_sink {highlights |= 4}
              highlights
            };
            // premultiply
            let modelview = inverse_cam * edge.transform;
            let direction:[f32;3] = (
              // find the screen coords of the two end points, use vector between them
              Into::<Matrix4<f32>>::into(perspective)
              .transform_point(modelview.transform_point(Point3::new(0.0,0.0,-1.0)))
              - Into::<Matrix4<f32>>::into(perspective)
              .transform_point(modelview.transform_point(Point3::new(0.0,0.0,0.0)))
            ).normalize().into();
            let modelview = modelview * Matrix4::from_nonuniform_scale(HL_EDGE_SCALE,HL_EDGE_SCALE,1.0);
            target.draw(
              verts, idxs,
              &highlight_program,
              &uniform! {
                modelview: Into::<[[f32;4];4]>::into(modelview),
                perspective: perspective,
                direction: direction,
                highlights: highlights,
                time: (new_time as f64 / 1_000_000_000.0) as f32,
              },
              &highlight_params
            ).unwrap();
          }
        }
      }
    }


    // draw text
    match graph.get_text(selected) {
      None => {},
      Some(item_text) => {
        let lines = item_text.len() as f32;
        for (i, line) in item_text.iter().enumerate() {
          let i = i as f32;
          let text = glium_text::TextDisplay::new(&text_system, &font, line);
          let text_height = TEXT_LINE_HEIGHT / height as f32;
          let line_size = text_height * 1.5;
          let y_start = -1.0 + text_height * 0.5 + line_size * (lines - 1.0);
          let text_transform = [
            [text_height * aspect_ratio, 0.0, 0.0, 0.0],
            [0.0, text_height, 0.0, 0.0],
            [0.0, 0.0, text_height, 0.0],
            [-1.0, y_start - i*line_size, 0.0, 1.0],
          ];
          glium_text::draw(&text, &text_system, &mut target, text_transform, (1.0, 1.0, 0.0, 1.0));
        }
      }
    }

    // Finish up and swap buffers
    target.finish().unwrap();

    // process events
    for event in display.poll_events() {
      match event {
        Event::Closed => { return; }
        Event::MouseInput(state,button) => {
          if state == ElementState::Pressed{ match button {
            MouseButton::Left => {graph.select_source(selected)},
            MouseButton::Right => {graph.select_sink(selected)},
            _ => {},
          }}
        }
        Event::MouseMoved(x,y) => {
          let x = std::cmp::min(width as i32, std::cmp::max(0,x)) as u32;
          let y = std::cmp::min(height as i32, std::cmp::max(0,(height as i32 - y))) as u32;
          //item picking
          let id = util::pick_from(&picker_buf, &picker_output, x, y);
          selected = id;
          graph.hover(id);
        }
        Event::KeyboardInput(state, _scode, vcode) => {
          if let Some(key) = vcode {
            if state == ElementState::Pressed { match key {
              VirtualKeyCode::LBracket => {
                _dev_multiplier /= 1.05;
              }
              VirtualKeyCode::RBracket => {
                _dev_multiplier *= 1.05;
              }
              // camera rotation
              VirtualKeyCode::A => {
                camera.transform = camera.transform.concat(&Decomposed{
                  scale: 1.0, disp: Vector3::zero(),
                  rot:Quaternion::from_angle_y(Deg(-TURN_SPEED)),
                }.into());
              }
              VirtualKeyCode::D => {
                camera.transform = camera.transform.concat(&Decomposed{
                  scale: 1.0, disp: Vector3::zero(),
                  rot:Quaternion::from_angle_y(Deg(TURN_SPEED)),
                }.into());
              }
              VirtualKeyCode::S => {
                camera.transform = camera.transform.concat(&Decomposed{
                  scale: 1.0, disp: Vector3::zero(),
                  rot:Quaternion::from_angle_x(Deg(TURN_SPEED)),
                }.into());
              }
              VirtualKeyCode::W => {
                camera.transform = camera.transform.concat(&Decomposed{
                  scale: 1.0, disp: Vector3::zero(),
                  rot:Quaternion::from_angle_x(Deg(-TURN_SPEED)),
                }.into());
              }
              VirtualKeyCode::R => {
                camera.transform = camera.transform.concat(&Decomposed{
                  scale: 1.0, disp: Vector3::zero(),
                  rot:Quaternion::from_angle_z(Deg(TURN_SPEED)),
                }.into());
              }
              VirtualKeyCode::F => {
                camera.transform = camera.transform.concat(&Decomposed{
                  scale: 1.0, disp: Vector3::zero(),
                  rot:Quaternion::from_angle_z(Deg(-TURN_SPEED)),
                }.into());
              }
              //camera movement
              VirtualKeyCode::Z => {
                camera.transform = camera.transform.concat(&Decomposed{
                  scale: 1.0, rot: Quaternion::one(),
                  disp: cgmath::vec3(0.0, 0.0,ZOOM_SCALE)
                }.into());
              }
              VirtualKeyCode::X => {
                camera.transform = camera.transform.concat(&Decomposed{
                  scale: 1.0, rot: Quaternion::one(),
                  disp: cgmath::vec3(0.0, 0.0,-ZOOM_SCALE)
                }.into());
              }
              VirtualKeyCode::Up => {
                camera.transform = camera.transform.concat(&Decomposed{
                  scale: 1.0, rot: Quaternion::one(),
                  disp: cgmath::vec3(0.0,MOVE_SPEED, 0.0)
                }.into());
              }
              VirtualKeyCode::Down => {
                camera.transform = camera.transform.concat(&Decomposed{
                  scale: 1.0, rot: Quaternion::one(),
                  disp: cgmath::vec3(0.0,-MOVE_SPEED, 0.0)
                }.into());
              }
              VirtualKeyCode::Left => {
                camera.transform = camera.transform.concat(&Decomposed{
                  scale: 1.0, rot: Quaternion::one(),
                  disp: cgmath::vec3(-MOVE_SPEED, 0.0,0.0)
                }.into());
              }
              VirtualKeyCode::Right => {
                camera.transform = camera.transform.concat(&Decomposed{
                  scale: 1.0, rot: Quaternion::one(),
                  disp: cgmath::vec3(MOVE_SPEED, 0.0,0.0)
                }.into());
              }
              // misc
              VirtualKeyCode::Escape => { return; }
              VirtualKeyCode::Q => { return; }
              VirtualKeyCode::M => { rotate_light = !rotate_light; }
              VirtualKeyCode::P => { _switch = !_switch; }
              VirtualKeyCode::I => { camera = Camera::default(); }
              _ => {}
            }}
          }
        },
        _e @ _ => {
          //println!("ignored: {:?}", _e)
        }
      }
    }
  }
}

struct DAGraph {
  nodes: Vec<DAGNode>,
  edges: Vec<DAGEdge>,
  source: Option<usize>,
  sink: Option<usize>,
  highlight: Option<usize>,
}

impl DAGraph {
  fn new() -> DAGraph {
    DAGraph {
      nodes: Vec::new(),
      edges: Vec::new(),
      highlight: None,
      source: None,
      sink: None,
    }
  }

  fn nodes_iter(&self) -> std::slice::Iter<DAGNode> {
    self.nodes.iter()
  }
  fn edges_iter(&self) -> std::slice::Iter<DAGEdge> {
    self.edges.iter()
  }

  fn add_node(&mut self,
    transform: Matrix4<f32>,
    material: Material,
    text: Vec<String>,
  ) {
    let id = self.nodes.len() * 2 + 1;
    self.nodes.push(DAGNode{
      transform: transform,
      id: id,
      in_edges: Vec::new(),
      out_edges: Vec::new(),
      visual: material,
      text: text,
      is_highlight: false,
      is_source: false,
      is_sink: false,
    });
  }

  fn add_edge(&mut self, from: usize, to: usize, material: Material, text: Vec<String>) {
    let index = self.edges.len();
    let id = index * 2 + 2;
    let f_node_loc = self.nodes[from].transform.transform_point(Point3::new(0.0,0.0,0.0));
    let t_node_loc = self.nodes[to].transform.transform_point(Point3::new(0.0,0.0,0.0));
    let f_to_t = f_node_loc - t_node_loc;
    let transform: Matrix4<f32> = 
      Decomposed {
        scale: 1.0, disp: f_node_loc.to_vec(),
        rot: Quaternion::from_arc(cgmath::vec3(0.0,0.0,1.0),f_to_t, None),
      }.into();
    let transform = transform.concat(&Matrix4::from_nonuniform_scale(1.0,1.0,f_to_t.magnitude()));
      // I wanted to simplfy the above by using look_at, but it doesn't work that way
      // Matrix4::from_translation(f_node_loc.to_vec());
      // Matrix4::from_scale((t_node_loc - f_node_loc).magnitude())
      // .concat(&Matrix4::look_at(
      //   f_node_loc, t_node_loc, cgmath::vec3(0.0,1.0,0.0)
      // ));
    self.edges.push(DAGEdge{
      id: id,
      from_node: from,
      to_node: to,
      visual: material,
      transform: transform,
      text: text,
      is_highlight: false,
      is_source: false,
      is_sink: false,
    });
    self.nodes[from].out_edges.push(index);
    self.nodes[to].in_edges.push(index);
  }

  fn get_text(&self, id: usize) -> Option<&Vec<String>> {
    if id == 0 { return None }
    if id % 2 == 1 {
      // node
      let index = (id - 1) / 2;
      Some(&self.nodes[index].text)
    } else {
      // edge
      let index = (id - 2) / 2;
      Some(&self.edges[index].text)
    }
  }

  fn hover(&mut self, id:usize){
    // clear old hovering
    if let Some(old_id) = self.highlight {
      if old_id == id { return }
      if old_id % 2 == 1 {
        // was hovering on a node
        let index = (old_id - 1) / 2;
        self.nodes[index].is_highlight = false;
      } else {
        // was hovering on an edge
        let index = (old_id - 2) / 2;
        self.edges[index].is_highlight = false;
      }
    }
    // set new hovering
    if id > 0 {
      self.highlight = Some(id);
      if id % 2 == 1 {
        // hover on a node
        let index = (id - 1) / 2;
        self.nodes[index].is_highlight = true;
      } else {
        // hover on an edge
        let index = (id - 2) / 2;
        self.edges[index].is_highlight = true;
      }
    } else {
      self.highlight = None;
    }
  }

  fn select_source(&mut self, id:usize){
    // clear old source
    if let Some(old_id) = self.source {
      if old_id == id { return }
      self.clear_sources();
    }
    // set new source
    if id > 0 {
      self.source = Some(id);
      if id % 2 == 1 {
        // select a node
        let index = (id - 1) / 2;
        self.make_source(index);
      } else {
        // select an edge
        let index = (id - 2) / 2;
        self.edges[index].is_source = true;
        let first_node = self.edges[index].to_node;
        self.make_source(first_node);
      }
    }
  }

  fn select_sink(&mut self, id:usize){
    // clear old sink
    if let Some(old_id) = self.sink {
      if old_id == id { return }
      self.clear_sinks();
    }
    // set new sink
    if id > 0 {
      self.sink = Some(id);
      if id % 2 == 1 {
        // select a node
        let index = (id - 1) / 2;
        self.make_sink(index);
      } else {
        // select an edge
        let index = (id - 2) / 2;
        self.edges[index].is_sink = true;
        let first_node = self.edges[index].from_node;
        self.make_sink(first_node);
      }
    }
  }

  fn make_source(&mut self, node: usize){
    self.nodes[node].is_source = true;
    let out_edges = self.nodes[node].out_edges.clone();
    for &edge in out_edges.iter() {
      if self.edges[edge].is_source == false {
        self.edges[edge].is_source = true;
        let nextnode = self.edges[edge].to_node;
        self.make_source(nextnode);
      }
    }
  }

  fn make_sink(&mut self, node: usize){
    self.nodes[node].is_sink = true;
    let in_edges = self.nodes[node].in_edges.clone();
    for &edge in in_edges.iter() {
      if self.edges[edge].is_sink == false {
        self.edges[edge].is_sink = true;
        let nextnode = self.edges[edge].from_node;
        self.make_sink(nextnode);
      }
    }
  }
  
  fn clear_sources(&mut self){
    self.source = None;
    for node in self.nodes.iter_mut() {
      node.is_source = false;
    }
    for edge in self.edges.iter_mut() {
      edge.is_source = false;
    }
  }

  fn clear_sinks(&mut self){
    self.sink = None;
    for node in self.nodes.iter_mut() {
      node.is_sink = false;
    }
    for edge in self.edges.iter_mut() {
      edge.is_sink = false;
    }
  }
}


struct DAGNode {
  transform: Matrix4<f32>,
  id: usize,
  in_edges: Vec<usize>,  // index in global table
  out_edges: Vec<usize>, // index in global table
  visual: Material,
  text: Vec<String>,
  is_highlight: bool,
  is_source: bool,
  is_sink: bool,
}

struct DAGEdge {
  id: usize,
  from_node: usize,  // index in global table
  to_node: usize,    // index in global table
  visual: Material,
  transform: Matrix4<f32>,
  text: Vec<String>,
  is_highlight: bool,
  is_source: bool,
  is_sink: bool,
}

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: (f32, f32, f32),
    pub normal: (f32,f32,f32),
    pub tex_coords: (f32,f32),
}
implement_vertex!(Vertex, position, normal, tex_coords);

// material for 3d display
struct Material {
  diffuse_color: [f32;3],
  specular_color: [f32;3],
  model: String,
}

impl Default for Material {
  fn default() -> Self {
    Material {
      diffuse_color: [0.5,0.5,0.5],
      specular_color: [0.8,0.8,0.8],
      model: "None".to_string(), 
    }
  }
}

// this represents a camera, its transform is inversed to get the
// scene transformation
struct Camera {
  fov: f32,
  transform: Matrix4<f32>,
}

impl Default for Camera {
  fn default() -> Self {
    Camera {
      fov: 3.14159 / 3.0,
      transform: Matrix4::look_at(
        [0.0,0.0,50.0].into(),
        [0.0,0.0,0.0].into(),
        [0.0,1.0,0.0].into(),
      ),
    }
  }
}