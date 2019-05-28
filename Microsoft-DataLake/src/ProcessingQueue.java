/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-data-lake-store-java/blob/master/src/main/java/com/microsoft/azure/datalake/store/ProcessingQueue.java
 *
 *  Created on: May 28, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.datalake.store.mirror;

import java.util.LinkedList;
import java.util.Queue;

/*
1. Caller can enqueue directories to process that will picked up by an open thread - add() method
2. Caller can dequeue directories when it is ready to process - poll() method
3. If the queue is empty, then caller blocks until an item becomes available in the queue - behaviour of poll()
4. Caller should indicate when it is done processing an item it popped from the queue - unregistered method
5. Caller should indicate when it has started processing an item
*/

class ProcessingQueue<T> {
    private Queue<T> internalQueue = new LinkedList<>();
    private int processorCount = 0;
    public synchronized void add(T item) {
        if (item == null) throw new IllegalArgumentException("Cannot put null into queue");
        internalQueue.add(item);
        this.notifyAll();
    }

    public synchronized T poll() {
        try {
            while (isQueueEmpty() && !done())
            {
                this.wait();
            }
            if (!isQueueEmpty())
            {
                processorCount++;
                return internalQueue.poll();
            }
            if (done())
            {
                return null;
            }
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
        }
        return null;
    }

    public synchronized void unregistered() 
    {
        processorCount--;
        if (processorCount < 0)
        {
            throw new IllegalStateException("too many unregister()'s. ProcessorCount is now " + processorCount);
        }
        if (done()) this.notifyAll();
    }

    private boolean done()
    {
        return (processorCount == 0 && isQueueEmpty());
    }

    private boolean isQueueEmpty()
    {
        return (internalQueue.peek() == null);
    }
}
