/**
 * Google - Big Data Technology
 *
 *  Created on: Feb 10, 2019
 *  Student (MIG Virtual Developer): Tung Dang
 */

package com.google.cloud.bigquery.mirror;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.base.Predicates.instanceOf;
// import static com.google.common.collect.CollectPreconditions.checkRemove;

import com.google.common.collect.AbstractIterator;
import com.google.api.client.util.Lists;
import com.google.common.annotations.Beta;
import com.google.common.annotations.GwtCompatible;
import com.google.common.annotations.GwtIncompatible;
import com.google.common.base.Function;
import com.google.common.base.Objects;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.Iterables;
import com.google.common.collect.UnmodifiableIterator;
import com.google.common.collect.UnmodifiableListIterator;
import com.google.common.primitives.Ints;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.rpc.RetryInfo;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.NoSuchElementException;
import java.util.PriorityQueue;
import java.util.Queue;

import javax.print.attribute.standard.RequestingUserName;

import org.apache.http.conn.ssl.TrustSelfSignedStrategy;
import org.checkerframework.checker.nullness.compatqual.NonNullDecl;
import org.checkerframework.checker.nullness.compatqual.NullableDecl;

@GwtCompatible(emulated = true)
public final class Iterators {
    private Iterators() {}

    static <T> UnmodifiableIterator<T> emptyIterator() {
        return emptyListIterator();
    }

    @SuppressWarnings("unchecked")
    static <T> UnmodifiableListIterator<T> emptyListIterator() {
        return (UnmodifiableListIterator<T>) ArrayItr.EMPTY;
    }

    private enum EmptyModifiableIterator implements Iterator<Object> {
        INSTANCE;

        @Override 
        public boolean hasNext() {
            return false;
        }

        @Override
        public Object next() {
            throw new NoSuchElementException();
        }

        @Override
        public void remove() {
            checkRemove(false);
        }
    }

    @SuppressWarnings("unchecked")
    static <T> Iterator<T> emptyModifiableIterator() {
        return (Iterator<T>) EmptyModifiableIterator.INSTANCE;
    }

    public static <T> UnmodifiableIterator<T> unmodifiableIterator (
        final Iterator<? extends T> iterator) 
    {
        checkNotNull(iterator);
        if (iterator instanceof UnmodifiableIterator)
        {
            @SuppressWarnings("unchecked")
            UnmodifiableIterator<T> result = (UnmodifiableIterator<T>) iterator;
            return result; 
        }
        return new UnmodifiableIterator<T>() 
        {
            @Override
            public boolean hasNext() 
            {
                return iterator.hasNext();
            }

            @Override
            public T next()
            {
                return iterator.next();
            }
        };
    }

    @Deprecated
    public static <T> UnmodifiableIterator<T> unmodifiableIterator(UnmodifiableIterator<T> iterator)
    {
        return checkNotNull(iterator);
    }

    public static int size(Iterator<?> iterator)
    {
        long count = 0L;
        while (iterator.hasNext())
        {
            iterator.next();
            count++;
        }
        return Ints.saturatedCast(count);
    }

    public static boolean contains(Iterator<?> iterator, @NullableDecl Object element)
    {
        if (element == null)
        {
            while (iterator.hasNext())
            {
                if (iterator.next() == null)
                {
                    return true;
                }
            }
        }
        else 
        {
            while (iterator.hasNext())
            {
                if (element.equals(iterator.next()))
                {
                    return true;
                }
            }
        }
        return false;
    }

    @CanIgnoreReturnValue
    public static boolean removeAll(Iterator<?> removeFrom, Collection<?> elementsToRemove)
    {
        checkNotNull(elementsToRemove);
        boolean result = false;
        while (removeFrom.hasNext())
        {
            if (elementsToRemove.contains(removeFrom.next()))
            {
                removeFrom.remove();
                result = true;
            }
        }
        return result;
    }

    @CanIgnoreReturnValue
    public static <T> boolean removeIf(Iterator<T> removeFrom, Predicate<? super T> predicate)
    {
        checkNotNull(predicate);
        boolean modified = false;
        while (removeFrom.hasNext())
        {
            if (predicate.apply(removeFrom.next()))
            {
                removeFrom.remove();
                modified = true;
            }
        }
        return modified;
    }

    @CanIgnoreReturnValue
    public static boolean retainAll(Iterator<?> removeFrom, Collection<?> elementsToRetain)
    {
        checkNotNull(elementsToRetain);
        boolean result = false;
        while (removeFrom.hasNext())
        {
            if (!elementsToRetain.contains(removeFrom.next()))
            {
                removeFrom.remove();
                result = true;
            }
        }
        return result;
    }

    public static boolean elementsEqual(Iterator<?> iterator1, Iterator<?> iterator2) 
    {
        while (iterator1.hasNext())
        {
            if (!iterator2.hasNext())
            {
                return false;
            }
            Object o1 = iterator1.next();
            Object o2 = iterator2.next();
            if (!Objects.equal(o1, o2))
            {
                return false;
            }
        }
        return !iterator2.hasNext();
    }

    public static String toString(Iterator<?> iterator)
    {
        StringBuilder sb = new StringBuilder().append('[');
        boolean first = true;
        while (iterator.hasNext())
        {
            if (!first)
            {
                sb.append(", ");
            }
            first = false;
            sb.append(iterator.next());
        }
        return sb.append(']').toString();
    }

    @CanIgnoreReturnValue
    public static <T> T getOnlyElement(Iterator<T> iterator)
    {
        T first = iterator.next();
        if (!iterator.hasNext())
        {
            return first;
        }

        StringBuilder sb = new StringBuilder().append("expected one element but was: <").append(first);
        for (int i = 0; i < 4 && iterator.hasNext(); i++)
        {
            sb.append(", ").append(iterator.next());
        }
        if (iterator.hasNext())
        {
            sb.append(", ...");
        }
        sb.append('>');
        throw new IllegalAccessException(sb.toString());
    }

    @CanIgnoreReturnValue
    @NullableDecl
    public static <T> T getOnlyElement(Iterator<? extends T> iterator, @NullableDecl T defaultValue)
    {
        return iterator.hasNext() ? getOnlyElement(iterator) : defaultValue;
    }

    @GwtIncompatible
    public static <T> T[] toArray(Iterator<? extends T> iterator, Class<T> type)
    {
        List<T> list = Lists.newArrayList(iterator);
        return Iterables.toArray(list, type);
    }

    @CanIgnoreReturnValue
    public static <T> boolean addAll(Collection<T> addTo, Iterator<? extends T> iterator)
    {
        checkNotNull(addTo);
        checkNotNull(iterator);
        boolean wasModified = false;
        while (iterator.hasNext())
        {
            wasModified |= addTo.add(iterator.next());
        }
        return wasModified;
    }

    public static int frequency(Iterator<?> iterator, @NullableDecl Object element) 
    {
        int count = 0;
        while (contains(iterator, element))
        {
            count++;
        }
        return count;
    }

    public static <T> Iterator<T> cycle(final Iterable<T> iterable) 
    {
        checkNotNull(iterable);
        return new Iterator<T>() {
            Iterator<T> iterator = emptyModifiableIterator();

            @Override
            public boolean hasNext()
            {
                return iterator.hasNext() || iterable.iterator().hasNext();
            }

            @Override
            public T next() 
            {
                if (!iterator.hasNext())
                {
                    iterator = iterable.iterator();
                    if (!iterator.hasNext())
                    {
                        throw new NoSuchElementException();
                    }
                }
                return iterator.next();
            }

            @Override
            public void remove()
            {
                iterator.remove();
            }
        };
    }

    @SafeVarargs
    public static <T> Iterator<T> cycle(T... elements) 
    {
        return cycle(Lists.newArrayList(elements));
    }

    /**
     * Returns an Iterator that walks the specified array, nulling out elements behind it.
     * This can avoid memory leaks when an element is no longer necessary.
     * This is mainly just to avoid the itermediate ArrayDeque in ConsummingQueueIterator 
     */
    private static <T> Iterator<T> consumingForArray(final T... elements)
    {
        return new UnmodifiableIterator<T>() {
            int index = 0;

            @Override
            public boolean hasNext()
            {
                return index < elements.length;
            }

            @Override
            public T next() 
            {
                if (!hasNext())
                {
                    throw new NoSuchElementException();
                }
                T result = elements[index];
                elements[index] = null;
                index++;
                return result;
            }
        };
    }

    public static <T> Iterator<T> concat(Iterator<? extends T> a, Iterator<? extends T> b)
    {
        checkNotNull(a);
        checkNotNull(b);
        return concat(consumingForArray(a,b));
    }

    public static <T> Iterator<T> concat(
        Iterator<? extends T>a, Iterator<? extends T>b, Iterator<? extends T>c) 
    {
        checkNotNull(a);
        checkNotNull(b);
        checkNotNull(c);
        return concat(consumingForArray(a,b,c));
    }

    public static <T> Iterator<T> concat(
      Iterator<? extends T> a,
      Iterator<? extends T> b,
      Iterator<? extends T> c,
      Iterator<? extends T> d) 
    {
        checkNotNull(a);
        checkNotNull(b);
        checkNotNull(c);
        checkNotNull(d);
        return concat(consumingForArray(a, b, c, d));
    }

    public static <T> Iterator<T> concat(Iterator<? extends T>... inputs)
    {
        return concatNoDefensiveCopy(Arrays.copyOf(inputs, inputs.length));
    }

    public static <T> Iterator<T> concat(Iterator<? extends Iterator<? extends T>> inputs)
    {
        return new ConcatenatedIterator<T>(inputs);
    }

    static <T> Iterator<T> concatNoDefensiveCopy(Iterator<? extends T>... inputs)
    {
        for (Iterator<? extends T> input : checkNotNull(inputs))
        {
            checkNotNull(input);
        }
        return concat(consumingForArray(inputs));
    }

    public static <T> UnmodifiableIterator<List<T>> partition(Iterator<T> iterator, int size)
    {
        return partitionImpl(iterator, size, false);
    }

    public static <T> UnmodifiableIterator<List<T>> paddedParitition(Iterator<T> iterator, int size)
    {
        return partitionImpl(iterator, size, true);
    }

    private static <T> UnmodifiableIterator<List<T>> partitionImpl(
        final Iterator<T> iterator, final int size, final boolean pad)
    {
        checkNotNull(iterator);
        checkArgument(size > 0);
        return new UnmodifiableIterator<List<T>>() {
            @Override
            public boolean hasNext()
            {
                return iterator.hasNext();
            }

            @Override
            public List<T> next()
            {
                if (!hasNext())
                {
                    throw new NoSuchElementException();
                }
                Object[] array = new Object[size];
                int count = 0;
                for (; count < size && iterator.hasNext(); count++)
                {
                    array[count] = iterator.next();
                }
                for (int i = count; i < size; i++)
                {
                    array[i] = null;
                }

                @SuppressWarnings("unchecked")
                List<T> list = Collections.unmodifiableList((List<T>) Arrays.asList(array));
                return (pad || count == size) ? list : list.subList(0, count);
            }
        };
    }

    public static <T> UnmodifiableIterator<T> filter (
        final Iterator<T> unfiltered, final Predicate<? super T> retainIfTrue)
    {
        checkNotNull(unfiltered);
        checkNotNull(retainIfTrue);
        return new AbstractIterator<T>()
        {
            @Override
            protected T computeNext()
            {
                while (unfiltered.hasNext())
                {
                    T element = unfiltered.next();
                    if (retainIfTrue.apply(element))
                    {
                        return element;
                    }
                }
                return endOfData();
            }
        };
    }

    @SuppressWarnings("unchecked")
    @GwtIncompatible
    public static <T> UnmodifiableIterator<T> filter(Iterator<?> unfiltered, Class<T> desiredType)
    {
        return (UnmodifiableIterator<T>) filter(unfiltered, instanceOf(desiredType));
    }

    public static <T> boolean any(Iterator<T> iterator, Predicate<? super T> predicate)
    {
        return indexOf(iterator, predicate) != -1;
    }

    public static <T> boolean all(Iterator<T> iterator, Predicate<? super T> predicate)
    {
        checkNotNull(predicate);
        while (iterator.hasNext())
        {
            T element = iterator.next();
            if (!predicate.apply(element))
            {
                return false;
            }
        }
        return true;
    }

    public static <T> T find(Iterator<T> iterator, Predicate<? super T> predicate)
    {
        checkNotNull(iterator);
        checkNotNull(predicate);
        while(iterator.hasNext())
        {
            T t = iterator.next();
            if (predicate.apply(t))
            {
                return t;
            }
        }
        throw new NoSuchElementException();
    }

    public static <T> Optional<T> tryFind(Iterator<T> iterator, Predicate<? super T> predicate)
    {
        checkNotNull(iterator);
        checkNotNull(predicate);
        while (iterator.hasNext())
        {
            T t = iterator.next();
            if (predicate.apply(t))
            {
                return Optional.of(t);
            }
        }
        return Optional.absent();
    }

    public static <T> int indexOf(Iterator<T> iterator, Predicate<? super T> predicate)
    {
        checkNotNull(predicate, "predicate");
        for (int i=0; iterator.hasNext(); i++)
        {
            T current = iterator.next();
            if (predicate.apply(current))
            {
                return i;
            }
        }
        return -1;
    }

    
}
