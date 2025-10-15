import axios from 'axios';
import * as _ from 'underscore';

const API_URL = process.env.NODE_ENV === 'development' ? 'http://localhost:7860/gradio_api' : '/gradio_api';

export const describeMaskWithoutStreaming = _.throttle(async (
  maskBase64: string,
  imageBase64: string,
  query: string
): Promise<string> => {
  try {
    const response = await axios.post(`${API_URL}/run/describe_without_streaming`, {
      data: [imageBase64, maskBase64, query],
    });
    
    console.log("response", response.data);
    return response.data.data[0];
  } catch (error) {
    console.error('Error describing mask:', error);
    throw error;
  }
}, 100);

export const describeMask = _.throttle(async (
  maskBase64: string,
  imageBase64: string,
  query: string,
  onStreamUpdate: (token: string) => void,
  onQueueUpdate?: (status: {
    inQueue: boolean,
    rank?: number,
    queueSize?: number,
    rankEta?: number | null
  }) => void
): Promise<string> => {
  console.log("describeMask");
  const initiateResponse = await axios.post(`${API_URL}/call/describe`, {
    data: [imageBase64, maskBase64, query],
  });
  
  const eventId = initiateResponse.data.event_id;
  
  const response = await axios.get(`${API_URL}/queue/data?session_hash=${eventId}`, {
    headers: {
      'Accept': 'text/event-stream',
    },
    responseType: 'stream',
    adapter: 'fetch',
  });

  const stream = response.data;
  const reader = stream.pipeThrough(new TextDecoderStream()).getReader();

  let result = '';
  let partialMessage = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      return result;
    }

    // Concatenate with any previous partial message
    const currentData = partialMessage + value;
    const lines = currentData.split('\n');
    
    // Save the last line if it's incomplete
    partialMessage = lines[lines.length - 1];
    
    // Process all complete lines except the last one
    let eventType = '';
    for (let i = 0; i < lines.length - 1; i++) {
      const line = lines[i];
      if (line.startsWith('event: ')) {
        eventType = line.slice(7); // Remove 'event: ' prefix
        console.log('Event message', line);
      } else if (line.startsWith('data: ')) {
        const eventData = line.slice(6); // Remove 'data: ' prefix
        try {
          let data = JSON.parse(eventData);
          if (data['msg']) {
            eventType = data['msg'];
            if (eventType === 'process_generating') {
              eventType = 'generating';
              data = data['output']['data'];
            } else if (eventType === 'process_completed') {
              eventType = 'complete';
              data = data['output']['data'];
            }
          }
          
          if (eventType === 'estimation' && onQueueUpdate) {
            onQueueUpdate({
              inQueue: true,
              rank: data.rank,
              queueSize: data.queue_size,
              rankEta: data.rank_eta
            });
          } else if (eventType === 'process_starts' && onQueueUpdate) {
            onQueueUpdate({
              inQueue: false
            });
          } else if ((eventType === 'generating' || eventType === 'complete') && data[0]) {
            result = data[0];
            onStreamUpdate(data[0]);
            
            if (eventType === 'complete') {
              return result;
            }
          }
        } catch (e) {
          console.log('Error parsing SSE message:', e);
        }
      } else if (line !== '') {
        console.log('Unknown message', line);
      }
    }
  }
}, 100);

export const imageToSamEmbedding = _.throttle(async (
  imageBase64: string,
  onQueueUpdate?: (status: {
    inQueue: boolean,
    rank?: number,
    queueSize?: number,
    rankEta?: number | null
  }) => void
): Promise<string> => {
  // First call to initiate the process
  const initiateResponse = await axios.post(`${API_URL}/call/image_to_sam_embedding`, {
    data: [imageBase64]
  });
  
  const eventId = initiateResponse.data.event_id;
  
  // Get the stream for queue updates and results
  const response = await axios.get(`${API_URL}/queue/data?session_hash=${eventId}`, {
    headers: {
      'Accept': 'text/event-stream',
    },
    responseType: 'stream',
    adapter: 'fetch',
  });

  const stream = response.data;
  const reader = stream.pipeThrough(new TextDecoderStream()).getReader();

  let result = '';
  let partialMessage = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      return result;
    }

    // Concatenate with any previous partial message
    const currentData = partialMessage + value;
    const lines = currentData.split('\n');
    
    // Save the last line if it's incomplete (doesn't end with \n)
    // The endpoint will send an empty line to indicate the end of a message, so it's ok to not process the partial message.
    partialMessage = lines[lines.length - 1];
    
    // Process all complete lines except the last one
    let eventType = '';
    for (let i = 0; i < lines.length - 1; i++) {
      const line = lines[i];
      if (line.startsWith('event: ')) {
        eventType = line.slice(7);
      } else if (line.startsWith('data: ')) {
        const eventData = line.slice(6);
        try {
          let data = JSON.parse(eventData);
          if (data['msg']) {
            eventType = data['msg'];
            console.log("Event type:", eventType);
            if (eventType === 'process_completed') {
              eventType = 'complete';
              data = data['output']['data'];
            }
          }
          
          if (eventType === 'estimation' && onQueueUpdate) {
            onQueueUpdate({
              inQueue: true,
              rank: data.rank,
              queueSize: data.queue_size,
              rankEta: data.rank_eta
            });
          } else if (eventType === 'process_starts' && onQueueUpdate) {
            onQueueUpdate({
              inQueue: false
            });
          } else if (eventType === 'complete' && data[0]) {
            result = data[0];
            console.log("Result for image to sam embedding:", result);
            return result;
          } else {
            console.log("Unknown event type:", eventType);
          }
        } catch (e) {
          console.log('Error parsing SSE message:', e, 'Raw data:', eventData);
        }
      }
    }
  }
}, 100);

export { API_URL };
