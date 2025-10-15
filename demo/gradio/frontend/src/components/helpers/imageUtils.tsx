import { Buffer } from 'buffer';

export const base64ToImage = async (base64String: string): Promise<HTMLImageElement> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = base64String.startsWith('data:') ? 
      base64String : 
      `data:image/png;base64,${base64String}`;
  });
};

export const imageToBase64 = (img: HTMLImageElement): string => {
  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d');
  ctx?.drawImage(img, 0, 0);
  return canvas.toDataURL('image/png');
}; 