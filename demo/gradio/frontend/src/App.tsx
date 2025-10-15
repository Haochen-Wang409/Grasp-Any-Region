// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { InferenceSession, Tensor } from "onnxruntime-web";
import React, { useContext, useEffect, useState, useRef } from "react";
import axios from "axios";
import "./assets/scss/App.scss";
import { handleImageScale } from "./components/helpers/scaleHelper";
import { modelScaleProps, QueueStatus } from "./components/helpers/Interfaces";
import { onnxMaskToImage, arrayToImageData, imageDataToURL } from "./components/helpers/maskUtils";
import { modelData } from "./components/helpers/onnxModelAPI";
import Stage, { DescriptionState } from "./components/Stage";
import AppContext from "./components/hooks/createContext";
import { imageToSamEmbedding } from "./services/maskApi";
import LoadingOverlay from "./components/LoadingOverlay";
import ErrorModal from './components/ErrorModal';
import QueueStatusIndicator from "./components/QueueStatusIndicator";

const ort = require("onnxruntime-web");

// Define image and model paths
const MODEL_DIR = "/model/sam_onnx_quantized_example.onnx";

const App = () => {
  const {
    clicks: [clicks, setClicks],
    image: [image, setImage],
    maskImg: [maskImg, setMaskImg],
    maskImgData: [maskImgData, setMaskImgData],
    isClicked: [isClicked, setIsClicked]
  } = useContext(AppContext)!;
  const [model, setModel] = useState<InferenceSession | null>(null);
  const [tensor, setTensor] = useState<Tensor | null>(null);
  const [modelScale, setModelScale] = useState<modelScaleProps | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [descriptionState, setDescriptionState] = useState<DescriptionState>({
    state: 'ready',
    description: ''
  });
  const [queueStatus, setQueueStatus] = useState<QueueStatus>({ inQueue: false });

  // Initialize the ONNX model
  useEffect(() => {
    const initModel = async () => {
      try {
        if (MODEL_DIR === undefined) return;
        const URL: string = MODEL_DIR;
        const model = await InferenceSession.create(URL);
        setModel(model);
      } catch (e) {
        console.log(e);
      }
    };
    initModel();
  }, []);

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const url = URL.createObjectURL(file);
      await loadImage(new URL(url));
    } catch (error) {
      setError('Failed to load image. Please try again with a different image.');
      console.error('Error loading image:', error);
    }
  };

  const loadImage = async (url: URL) => {
    try {
      setIsLoading(true);
      const img = new Image();
      img.src = url.href;
      img.onload = async () => {
        const { height, width, samScale } = handleImageScale(img);
        setModelScale({
          height: height,
          width: width,
          samScale: samScale,
        });
        img.width = width;
        img.height = height;
        setImage(img);

        // After image is loaded, fetch its embedding from Gradio
        await fetchImageEmbedding(img);
        setIsLoading(false);
      };
    } catch (error) {
      console.log(error);
      setIsLoading(false);
    }
  };

  const fetchImageEmbedding = async (img: HTMLImageElement) => {
    try {
      // Create a canvas to convert the image to base64
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      ctx?.drawImage(img, 0, 0);
      
      // Convert image to base64 data URL and extract the base64 string
      const base64Image = canvas.toDataURL('image/jpeg').split(',')[1];

      // Make request to Gradio API
      const samEmbedding = await imageToSamEmbedding(
        base64Image,
        (status: QueueStatus) => {
          setQueueStatus(status);
        }
      );

      // Convert base64 embedding back to array buffer
      const binaryString = window.atob(samEmbedding);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      // Create tensor from the embedding
      const embedding = new ort.Tensor(
        'float32',
        new Float32Array(bytes.buffer),  // Convert to Float32Array
        [1, 256, 64, 64] // SAM embedding shape
      );
      setTensor(embedding);
    } catch (error) {
      setQueueStatus({ inQueue: false }); // Reset queue status on error
      let errorMessage = 'Failed to process image. Please try again.';
      if (axios.isAxiosError(error)) {
        errorMessage = error.response?.data?.message || errorMessage;
      }
      setError(errorMessage);
      console.error('Error fetching embedding:', error);
    }
  };

  useEffect(() => {
    const handleMaskUpdate = async () => {
      await runONNX();
    };
    handleMaskUpdate();
  }, [clicks]);

  const runONNX = async () => {
    try {
      // Don't run if already described or is describing
      if (descriptionState.state !== 'ready') return;
      
      console.log('Running ONNX model with:', {
        modelLoaded: model !== null,
        hasClicks: clicks !== null,
        hasTensor: tensor !== null,
        hasModelScale: modelScale !== null
      });

      if (
        model === null ||
        clicks === null ||
        tensor === null ||
        modelScale === null
      ) {
        console.log('Missing required inputs, returning early');
        return;
      }
      else {
        console.log('Preparing model feeds with:', {
          clicks,
          tensorShape: tensor.dims,
          modelScale
        });

        const feeds = modelData({
          clicks,
          tensor,
          modelScale,
        });

        if (feeds === undefined) {
          console.log('Model feeds undefined, returning early');
          return;
        }

        console.log('Running model with feeds:', feeds);
        const results = await model.run(feeds);
        console.log('Model run complete, got results:', results);

        const output = results[model.outputNames[0]];
        console.log('Processing output with dims:', output.dims);

        // Calculate and log the mask area (number of non-zero values)
        const maskArray = Array.from(output.data as Uint8Array);
        const maskArea = maskArray.filter(val => val > 0).length;
        console.log('Mask area (number of non-zero pixels):', maskArea);

        // Double check that the state is ready before processing the mask since the state may have changed
        if (descriptionState.state !== 'ready') return;
        // If clicked, we only handle the first mask (note that mask will be cleared after clicking before handling to let us know if it's the first mask).
        if (isClicked && maskImgData != null) return;
        if (maskArea > 0) {
          setMaskImg(onnxMaskToImage(output.data, output.dims[2], output.dims[3], false));
          setMaskImgData(imageDataToURL(arrayToImageData(output.data, output.dims[2], output.dims[3], true)));
        } else {
          console.warn('No mask area detected, clearing mask');
          setMaskImg(null);
          // setMaskImgData(null);
        }
        
        console.log('Mask processing complete');
      }
    } catch (e) {
      setError('Failed to process the image. Please try again.');
      console.error('Error running ONNX model:', e);
    }
  };

  const handleNewRegion = () => {
    setDescriptionState({
      state: 'ready',
      description: ''
    } as DescriptionState);
    setMaskImg(null);
    // setMaskImgData(null);
    setIsClicked(false);
  };

  const handleCopyDescription = () => {
    navigator.clipboard.writeText(descriptionState.description);
  };

  const handleReset = () => {
    // Clear all states
    setDescriptionState({
      state: 'ready',
      description: ''
    } as DescriptionState);
    setMaskImg(null);
    // setMaskImgData(null);
    setImage(null);
    setClicks(null);
    setIsClicked(false);
  };

  return (
    <div className="flex flex-col h-screen">
      {isLoading && <LoadingOverlay />}
      {error && <ErrorModal message={error} onClose={() => setError(null)} />}
      <QueueStatusIndicator queueStatus={queueStatus} />
      <div className="flex-1">
        <Stage 
          onImageUpload={handleImageUpload} 
          descriptionState={descriptionState}
          setDescriptionState={setDescriptionState}
          queueStatus={queueStatus}
          setQueueStatus={setQueueStatus}
        />
      </div>
      <div className="description-container">
        <div className={`description-box ${descriptionState.state !== 'described' ? descriptionState.state : ''}`}>
          {descriptionState.description ? (
            descriptionState.description + (descriptionState.state === 'describing' ? '...' : '')
          ) : descriptionState.state === 'describing' ? (
            <em>Describing the region... (this may take a while if compute resources are busy)</em>
          ) : (
            image ? (
              <em>Click on the image to describe the region</em>
            ) : (
              <em>Upload an image to describe the region</em>
            )
          )}
        </div>
        <div className="description-controls">
          <button 
            onClick={handleCopyDescription}
            disabled={descriptionState.state !== 'described'}
          >
            Copy description
          </button>
          <button 
            onClick={handleNewRegion}
            disabled={descriptionState.state !== 'described'}
          >
            Describe a new region
          </button>
          <button 
            onClick={handleReset}
            className="reset-button"
            disabled={descriptionState.state === 'describing' || !image}
          >
            Try a new image
          </button>
        </div>
      </div>
    </div>
  );
};

export default App;
