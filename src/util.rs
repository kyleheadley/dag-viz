#![allow(dead_code)]

pub fn perspective(fov: f32, asp: f32, far: f32) -> [[f32;4];4] {
  let near = 0.1;
  let f = 1.0 / (fov / 2.0).tan();

  [
    [f * asp, 0.0, 0.0                       , 0.0],
    [0.0    ,  f , 0.0                       , 0.0],
    [0.0    , 0.0, (far+near)/(far-near)     , 1.0],
    [0.0    , 0.0, -(2.0*far*near)/(far-near), 0.0],
  ]
}

use std::mem;

pub fn uniform_of_id(id: usize) -> [u32;4] {
  let fixed_id = id as u32;
  let id_comp:[u8;4] = unsafe {mem::transmute(fixed_id)};
  let id_pad:[u32;4] = [
    id_comp[0] as u32,
    id_comp[1] as u32,
    id_comp[2] as u32,
    id_comp[3] as u32
  ];
  id_pad
}

use glium::texture::Texture2d;
use glium::framebuffer::SimpleFrameBuffer;
use glium::uniforms::MagnifySamplerFilter;
use glium::{Surface, Rect, BlitTarget};

pub fn pick_from(buff: &SimpleFrameBuffer, temp_tex: &Texture2d, x:u32, y:u32) -> usize {
  buff.blit_color(
    &Rect{left:x, bottom:y, width:1, height:1},
    &temp_tex.as_surface(),
    &BlitTarget{left:0, bottom:0, width:1, height:1},
    MagnifySamplerFilter::Nearest,
  );
  let data: Vec<Vec<(u8,u8,u8,u8)>> = temp_tex.read();
  let id:u32 = unsafe{mem::transmute(data[0][0])};
  id as usize
}

use glium::{backend,texture};
use image;
use std::io::{BufRead,Seek};

pub fn tex2d_load
  <F: backend::Facade, D: BufRead + Seek>
  (display: &F, data: D, filetype: image::ImageFormat) -> texture::Texture2d
{
  let pic = image::load(data,filetype).unwrap().to_rgba();
  let pic_dimensions = pic.dimensions();
  let raw_tex = texture::RawImage2d::from_raw_rgba_reversed(pic.into_raw(), pic_dimensions);
  let tex = texture::Texture2d::new(display, raw_tex).unwrap();
  tex
}
