/**
 * Google - Big Data Technology
 * https://github.com/googleapis/google-cloud-java/blob/master/google-cloud-clients/google-cloud-bigquery/src/main/java/com/google/cloud/bigquery/JobConfiguration.java
 *
 *  Created on: Feb 20, 2019
 *  Student (MIG Virtual Developer): Tung Dang
 */

package com.google.cloud.bigquery.mirror;

import static com.google.common.base.Preconditions.checkNotNull;
import com.google.common.base.MoreObjects;
import com.google.common.base.MoreObjects.ToStringHelper;
import java.io.Serializable;
import java.util.Objects;

public abstract class JobConfiguration implements Serializable 
{
    private static final long serialVersionUID = -548132177415406526L;
    private final Type type;
    public enum Type 
    {
        COPY,
        EXTRACT,
        LOAD,
        QUERY
    }

    public abstract static class Builder<T extends JobConfiguration, B extends Builder<T, B>>
    {
        private Type type;
        Builder(Type type)
        {
            this.type = checkNotNull(type);
        }

        @SuppressWarnings("unchecked")
        B self() 
        {
            return (B) this;
        }

        B setType(Type type)
        {
            this.type = checkNotNull(type);
            return self();
        }

        public abstract T build();
    }

    
}
