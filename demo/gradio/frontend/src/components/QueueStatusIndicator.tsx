import React from 'react';
import { QueueStatus } from './helpers/Interfaces';

interface QueueStatusIndicatorProps {
  queueStatus: QueueStatus;
}

const QueueStatusIndicator: React.FC<QueueStatusIndicatorProps> = ({ queueStatus }) => {
  if (!queueStatus.inQueue) return null;

  return (
    <div className="fixed top-4 right-4 bg-white rounded-lg shadow-lg p-4 z-50">
      <div className="flex flex-col gap-2">
        {queueStatus.rank === 0 ? (
          <p className="text-sm">You're next in line! ({queueStatus.queueSize} total in queue)</p>
        ) : (
          <p className="text-sm">Queue position: {queueStatus.rank! + 1} of {queueStatus.queueSize}</p>
        )}
        {queueStatus.rankEta && (
          <p className="text-sm text-gray-600">
            Estimated wait: {Math.ceil(queueStatus.rankEta)} seconds
          </p>
        )}
      </div>
    </div>
  );
};

export default QueueStatusIndicator; 