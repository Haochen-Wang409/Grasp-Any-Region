// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// Convert the onnx model mask prediction to ImageData
function arrayToImageData(input: any, width: number, height: number, binary: boolean) {
  let [r, g, b, a] = [0, 114, 189, 255]; // the masks's blue color
  let [r_bg, g_bg, b_bg, a_bg] = [0, 0, 0, 0]; // the background's white color
  if (binary) {
    [r, g, b, a] = [255, 255, 255, 255]; // black and white
    [r_bg, g_bg, b_bg, a_bg] = [0, 0, 0, 255]; // black and white
  }
  
  const arr = new Uint8ClampedArray(4 * width * height).fill(0);
  for (let i = 0; i < input.length; i++) {

    // Threshold the onnx model mask prediction at 0.0
    // This is equivalent to thresholding the mask using predictor.model.mask_threshold
    // in python
    if (input[i] > 0.0) {
      arr[4 * i + 0] = r;
      arr[4 * i + 1] = g;
      arr[4 * i + 2] = b;
      arr[4 * i + 3] = a;
    } else if (binary){
      arr[4 * i + 0] = r_bg;
      arr[4 * i + 1] = g_bg;
      arr[4 * i + 2] = b_bg;
      arr[4 * i + 3] = a_bg;
    }
  }
  return new ImageData(arr, height, width);
}

// Use a Canvas element to produce an image from ImageData
function imageDataToImage(imageData: ImageData) {
  const canvas = imageDataToCanvas(imageData);
  const image = new Image();
  image.src = canvas.toDataURL();
  return image;
}

function imageDataToURL(imageData: ImageData) {
  const canvas = imageDataToCanvas(imageData);
  return canvas.toDataURL();
}

// Canvas elements can be created from ImageData
function imageDataToCanvas(imageData: ImageData) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  ctx?.putImageData(imageData, 0, 0);
  return canvas;
}

// Convert the onnx model mask output to an HTMLImageElement
function onnxMaskToImage(input: any, width: number, height: number, binary: boolean) {
  return imageDataToImage(arrayToImageData(input, width, height, binary));
}

export { arrayToImageData, imageDataToImage, onnxMaskToImage, imageDataToURL };
