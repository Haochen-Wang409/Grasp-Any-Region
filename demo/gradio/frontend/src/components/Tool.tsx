import React, { useContext, useEffect, useState } from "react";
import AppContext from "./hooks/createContext";
import { ToolProps, QueueStatus } from "./helpers/Interfaces";
import * as _ from "underscore";
import { describeMask, describeMaskWithoutStreaming } from "../services/maskApi";
import ErrorModal from './ErrorModal';
import { DescriptionState } from "./Stage";

const prompt = "<image>\nDescribe the masked region in detail.";

const Tool = ({ 
  handleMouseMove, 
  descriptionState, 
  setDescriptionState,
  queueStatus,
  setQueueStatus 
}: ToolProps) => {
  console.log("Tool handleMouseMove");
  const {
    image: [image],
    maskImg: [maskImg, setMaskImg],
    maskImgData: [maskImgData, setMaskImgData],
    isClicked: [isClicked, setIsClicked]
  } = useContext(AppContext)!;

  const [shouldFitToWidth, setShouldFitToWidth] = useState(true);
  const bodyEl = document.body;
  const fitToPage = () => {
    if (!image) return;
    const maxWidth = window.innerWidth - 64; // Account for padding (32px on each side)
    const maxHeight = window.innerHeight - 200; // Account for header and some padding
    const imageAspectRatio = image.width / image.height;
    const containerAspectRatio = maxWidth / maxHeight;
    
    setShouldFitToWidth(
      imageAspectRatio > containerAspectRatio || 
      image.width > maxWidth
    );
  };
  const resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) {
      if (entry.target === bodyEl) {
        fitToPage();
      }
    }
  });
  useEffect(() => {
    fitToPage();
    resizeObserver.observe(bodyEl);
    return () => {
      resizeObserver.unobserve(bodyEl);
    };
  }, [image]);

  const imageClasses = "";
  const maskImageClasses = `absolute opacity-40 pointer-events-none`;

  const [error, setError] = useState<string | null>(null);
  const [useStreaming, setUseStreaming] = useState(true);

  useEffect(() => {
      if (!isClicked || !maskImg || !maskImgData || !image || descriptionState.state !== 'ready') {
        console.log("Not ready to call model, isClicked:", isClicked, "maskImg:", maskImg !== null, "maskImgData:", maskImgData !== null, "image:", image !== null, "descriptionState.state:", descriptionState.state);
        return;
      }

      try {
        setDescriptionState({
          state: 'describing',
          description: ''
        } as DescriptionState);

        const canvas = document.createElement('canvas');
        canvas.width = image.width;
        canvas.height = image.height;
        const ctx = canvas.getContext('2d');
        ctx?.drawImage(image, 0, 0);
        const imageBase64 = canvas.toDataURL('image/jpeg').split(',')[1];
        const maskBase64 = maskImgData.split(',')[1];

        const describeMaskWithFallback = async (useStreamingInFunction: boolean) => {
          try {
            let result;
            console.log("useStreaming", useStreaming, "useStreamingInFunction", useStreamingInFunction);
            if (useStreamingInFunction) {
              result = await describeMask(
                maskBase64, 
                imageBase64,
                prompt,
                (streamResult: string) => {
                  setDescriptionState({
                    state: 'describing',
                    description: streamResult
                  } as DescriptionState);
                },
                (status: QueueStatus) => {
                  setQueueStatus(status);
                }
              );
            } else {
              result = await describeMaskWithoutStreaming(
                maskBase64,
                imageBase64,
                prompt
              );
            }
            
            setDescriptionState({
              state: 'described',
              description: result
            } as DescriptionState);
            setQueueStatus({ inQueue: false });
            setIsClicked(false);
          } catch (error) {
            if (useStreaming) {
              console.log("Error describing mask, switching to non-streaming", error);
              setUseStreaming(false);
              describeMaskWithFallback(false);
            } else {
              setError('Failed to generate description. Please try again.');
              setDescriptionState({
                state: 'ready',
                description: ''
              } as DescriptionState);
              setIsClicked(false);
              console.error('Failed to describe mask:', error);
            }
          }
        };

        describeMaskWithFallback(useStreaming);

      } catch (error) {
        setIsClicked(false);
        setError('Failed to generate description. Please try again.');
        setDescriptionState({
          state: 'ready',
          description: ''
        } as DescriptionState);
        console.error('Failed to describe mask:', error);
      }
  }, [maskImgData]);

  const handleClick = async (e: React.MouseEvent<HTMLImageElement>) => {
    if (descriptionState.state !== 'ready') return;
    
    setMaskImg(null);
    setMaskImgData(null);
    setIsClicked(true);
    handleMouseMove(e);
  };

  return (
    <>
      {error && <ErrorModal message={error} onClose={() => setError(null)} />}
      <div className="relative flex items-center justify-center w-full h-full">
        {image && (
          <img
            onMouseMove={handleMouseMove}
            onMouseLeave={() => _.defer(() => (descriptionState.state === 'ready' && !isClicked) ? setMaskImg(null) : undefined)}
            onTouchStart={handleMouseMove}
            onClick={handleClick}
            src={image.src}
            className={`${
              shouldFitToWidth ? "w-full" : "h-full"
            } ${imageClasses} object-contain max-h-full max-w-full`}
          ></img>
        )}
        {maskImg && (
          <img
            src={maskImg.src}
            className={`${
              shouldFitToWidth ? "w-full" : "h-full"
            } ${maskImageClasses} object-contain max-h-full max-w-full`}
          ></img>
        )}
      </div>
    </>
  );
};

export default Tool;
