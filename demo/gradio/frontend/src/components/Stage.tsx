// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import React, { useContext, useState, useEffect } from "react";
import * as _ from "underscore";
import Tool from "./Tool";
import { modelInputProps, QueueStatus } from "./helpers/Interfaces";
import AppContext from "./hooks/createContext";
// import { describeMask } from '../services/maskApi';

interface DescriptionState {
  state: string; // 'ready', 'describing', 'described'
  description: string;
}

interface StageProps {
  onImageUpload: (event: React.ChangeEvent<HTMLInputElement>) => Promise<void>;
  descriptionState: DescriptionState;
  setDescriptionState: React.Dispatch<React.SetStateAction<DescriptionState>>;
  queueStatus: QueueStatus;
  setQueueStatus: (status: QueueStatus) => void;
}

const EXAMPLE_IMAGES = Array.from({ length: 21 }, (_, i) => `/examples/${i + 1}.jpg`);
const BREAKPOINT_MEDIUM = 2100;
const BREAKPOINT_SMALL = 1100;

const Stage = ({ onImageUpload, descriptionState, setDescriptionState, queueStatus, setQueueStatus }: StageProps) => {
  const {
    clicks: [, setClicks],
    image: [image],
    maskImg: [maskImg],
    maskImgData: [maskImgData]
  } = useContext(AppContext)!;

  const [isDragging, setIsDragging] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [imagesPerPage, setImagesPerPage] = useState(8);

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < BREAKPOINT_SMALL) {
        setImagesPerPage(1);
      } else if (window.innerWidth < BREAKPOINT_MEDIUM) {
        setImagesPerPage(4);
      } else {
        setImagesPerPage(8);
      }
    };

    // Set initial value
    handleResize();

    // Add event listener
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const getClick = (x: number, y: number): modelInputProps => {
    const clickType = 1;
    return { x, y, clickType };
  };

  const handleMouseMove = _.throttle((e: any) => {
    if (descriptionState.state !== 'ready') return;
    if (e.clientX === undefined || e.clientY === undefined) {
      console.warn('Mouse move event does not contain clientX or clientY');
      return;
    }
    let el = e.nativeEvent.target;
    const rect = el.getBoundingClientRect();
    
    // Calculate the actual dimensions of the contained image
    const containerAspectRatio = el.offsetWidth / el.offsetHeight;
    const imageAspectRatio = image ? image.width / image.height : 1;
    
    let renderedWidth, renderedHeight;
    if (containerAspectRatio > imageAspectRatio) {
      // Image is constrained by height
      renderedHeight = el.offsetHeight;
      renderedWidth = renderedHeight * imageAspectRatio;
    } else {
      // Image is constrained by width
      renderedWidth = el.offsetWidth;
      renderedHeight = renderedWidth / imageAspectRatio;
    }

    // Calculate the empty space offset
    const offsetX = (el.offsetWidth - renderedWidth) / 2;
    const offsetY = (el.offsetHeight - renderedHeight) / 2;

    // Get click position relative to the actual image
    let x = e.clientX - rect.left - offsetX;
    let y = e.clientY - rect.top - offsetY;

    // Convert to original image coordinates
    const scaleX = image ? image.width / renderedWidth : 1;
    const scaleY = image ? image.height / renderedHeight : 1;
    x *= scaleX;
    y *= scaleY;

    // Ensure coordinates are within bounds
    if (image) {
      x = Math.max(0, Math.min(x, image.width));
      y = Math.max(0, Math.min(y, image.height));
    }

    const click = getClick(x, y);
    if (click) {
      setClicks([click]);
    }
  }, 15);

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      const file = files[0];
      // Cast to unknown first, then to the desired type
      const syntheticEvent = {
        target: {
          files: [file]
        }
      } as unknown as React.ChangeEvent<HTMLInputElement>;
      
      onImageUpload(syntheticEvent);
    }
  };

  const flexCenterClasses = "flex items-center justify-center";
  
  // const handleDescribeMask = async () => {
  //   if (!maskImg || !maskImgData || !image) {
  //     console.warn('No mask or image available to describe');
  //     return;
  //   }

  //   try {
  //     const canvas = document.createElement('canvas');
  //     canvas.width = image.width;
  //     canvas.height = image.height;
  //     const ctx = canvas.getContext('2d');
  //     ctx?.drawImage(image, 0, 0);
  //     const imageBase64 = canvas.toDataURL('image/jpeg').split(',')[1];
  //     const maskBase64 = maskImgData.split(',')[1];

  //     const result = await describeMask(maskBase64, imageBase64);
  //     console.log('Mask description:', result.description);
      
  //     alert("Mask description: " + result.description);
  //   } catch (error) {
  //     console.error('Failed to describe mask:', error);
  //   }
  // };

  return (
    <div 
      className={`flex flex-col w-full h-full relative`}
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Title and Description */}
      <div className="w-full px-8 mb-8 flex flex-col justify-center mt-4">
        <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
          <h1 className="text-3xl font-bold text-center sm:text-left"><a href="/">Describe Anything Model Demo</a></h1>
          <div className="flex flex-wrap justify-center gap-4 sm:space-x-8 text-lg font-medium">
            <a href="https://describe-anything.github.io/" target="_blank" rel="noopener noreferrer" className="text-gray-600 hover:text-gray-800">Project Page</a>
            <a href="https://github.com/NVlabs/describe-anything?tab=readme-ov-file#simple-gradio-demo-for-detailed-localized-video-descriptions" target="_blank" rel="noopener noreferrer" className="text-gray-600 hover:text-gray-800">DAM for video</a>
          </div>
        </div>
        <div className="border-b border-gray-300 mt-4 mb-4"></div>
        {!image && <div className="space-y-4 text-gray-600 text-left">
          <p>Describe Anything Model (DAM) takes in a region of an image or a video in the form of points/boxes/scribbles/masks and outputs detailed descriptions to the region. For videos, it is sufficient to supply annotation on any frame.</p>
          <p>This demo supports DAM model that takes points on images as queries. For other use cases, please refer to the <a href="" className="text-gray-600 hover:text-gray-800 underline">inference scripts and video demo</a> for more details.</p>
        </div>}
      </div>

      {/* Main Content Area */}
      <div className={`flex items-center justify-center flex-grow overflow-hidden`}>
        {/* Main Stage */}
        <div 
          className={`${flexCenterClasses} relative w-full h-full max-h-[calc(100vh-300px)] px-8 ${
            isDragging ? 'border-4 border-dashed border-blue-500 bg-blue-50' : ''
          }`}
        >
          {image ? (
            <>
              <Tool 
                handleMouseMove={handleMouseMove} 
                descriptionState={descriptionState}
                setDescriptionState={setDescriptionState}
                queueStatus={queueStatus}
                setQueueStatus={setQueueStatus}
              />
            </>
          ) : (
            <>
              <div className="flex flex-col items-center gap-6 w-full h-full">
                <div className="flex-1" />
                
                <div className="text-gray-500 text-lg">
                  {isDragging ? 'Drop image here' : 'Upload your own image'}
                </div>
                <div className="flex gap-4 mb-8">
                  <input
                    type="file"
                    id="imageUpload"
                    accept="image/*"
                    onChange={onImageUpload}
                    className="hidden"
                  />
                  <label
                    htmlFor="imageUpload"
                    className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded cursor-pointer"
                  >
                    Upload Image
                  </label>
                </div>

                <div className="text-gray-500 text-lg">
                  or choose an example image below
                </div>
                
                <div className="relative w-full max-w-[2200px]">
                  {/* Left Arrow */}
                  <button
                    onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                    disabled={currentPage === 1}
                    className={`absolute left-0 top-1/2 -translate-y-1/2 z-10 p-4 ${
                      currentPage === 1 
                        ? 'text-gray-300 cursor-not-allowed' 
                        : 'text-gray-600 hover:text-gray-800'
                    }`}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                    </svg>
                  </button>

                  {/* Example Images */}
                  <div className="flex flex-wrap justify-center gap-8 px-16">
                    {EXAMPLE_IMAGES.slice(
                      (currentPage - 1) * imagesPerPage,
                      currentPage * imagesPerPage
                    ).map((src, index) => (
                      <img
                        key={index}
                        src={src}
                        alt={`Example ${index + 1}`}
                        className="w-[200px] h-[150px] object-cover rounded-sm cursor-pointer hover:opacity-80 transition-opacity"
                        onClick={() => {
                          fetch(src)
                            .then(res => res.blob())
                            .then(blob => {
                              const file = new File([blob], `example-${index + 1}.jpg`, { type: 'image/jpeg' });
                              const syntheticEvent = {
                                target: {
                                  files: [file]
                                }
                              } as unknown as React.ChangeEvent<HTMLInputElement>;
                              
                              onImageUpload(syntheticEvent);
                            });
                        }}
                      />
                    ))}
                  </div>

                  {/* Right Arrow */}
                  <button
                    onClick={() => setCurrentPage(prev => Math.min(prev + 1, Math.ceil(EXAMPLE_IMAGES.length / imagesPerPage)))}
                    disabled={currentPage === Math.ceil(EXAMPLE_IMAGES.length / imagesPerPage)}
                    className={`absolute right-0 top-1/2 -translate-y-1/2 z-10 p-4 ${
                      currentPage === Math.ceil(EXAMPLE_IMAGES.length / imagesPerPage)
                        ? 'text-gray-300 cursor-not-allowed'
                        : 'text-gray-600 hover:text-gray-800'
                    }`}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </button>

                  {/* Page Indicator */}
                  {/* <div className="w-full text-center mt-4 text-gray-600">
                    Page {currentPage} of {Math.ceil(EXAMPLE_IMAGES.length / imagesPerPage)}
                  </div> */}
                </div>

                <div className="flex-1" /> {/* Bottom spacer */}
                {/* Image Credits */}
                {!image && (
                <div className="pl-5 pr-5 text-gray-500 text-sm">
                  Image credit for example images: {' '}
                  <a 
                    href="https://segment-anything.com/terms" 
                    target="_blank" 
                    className="text-gray-600 hover:text-gray-800 underline"
                  >
                    Segment Anything Materials
                  </a>
                  {' '}(CC BY-SA 4.0)
                </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
      
    </div>
  );
};

export default Stage;
export type { DescriptionState };
