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

import com.google.common.annotations.Beta;
import com.google.common.annotations.GwtCompatible;
import com.google.common.annotations.GwtIncompatible;
import com.google.common.base.Function;
import com.google.common.base.Objects;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.UnmodifiableIterator;
import com.google.common.collect.UnmodifiableListIterator;
import com.google.common.primitives.Ints;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
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
import org.checkerframework.checker.nullness.compatqual.NonNullDecl;

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
}