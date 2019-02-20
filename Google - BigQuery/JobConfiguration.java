/**
 * Google - Big Data Technology
 * https://github.com/googleapis/google-cloud-java/blob/master/google-cloud-clients/google-cloud-bigquery/src/main/java/com/google/cloud/bigquery/JobConfiguration.java
 *
 *  Created on: Feb 20, 2019
 *  Student (MIG Virtual Developer): Tung Dang
 */

package com.google.cloud.bigquery.mirror;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.cloud.bigquery.CopyJobConfiguration;
import com.google.cloud.bigquery.ExtractJobConfiguration;
import com.google.cloud.bigquery.LoadConfiguration;
import com.google.cloud.bigquery.QueryJobConfiguration;
import com.google.common.base.MoreObjects;
import com.google.common.base.MoreObjects.ToStringHelper;
import com.google.rpc.RetryInfo;

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

    JobConfiguration(Builder builder)
    {
        this.type = builder.type;
    }

    public Type getType()
    {
        return type;
    }

    public abstract Builder toBuilder();

    ToStringHelper toStringHelper() 
    {
        return MoreObjects.toStringHelper(this).add("Type", type);
    }

    @Override
    public String toString() 
    {
        return toStringHelper().toString();
    }

    final int baseHashCode()
    {
        return Objects.hash(type);
    }

    final boolean baseEquals(JobConfiguration jobConfiguration)
    {
        return Objects.equals(toPb(), jobConfiguration.toPb());
    }

    abstract JobConfiguration setProjectId(String projectId);
    abstract com.google.api.services.bigquery.model.JobConfiguration toPb();
    @SuppressWarnings("unchecked")
    static<T extends JobConfiguration> T fromPb(
        com.google.api.services.bigquery.model.JobConfiguration configurationPb) {
        if (configurationPb.getCopy() != null)
        {
            return (T) CopyJobConfiguration.fromPb(configurationPb);
        } else if (configurationPb.getExtract() != null) {
            return (T) ExtractJobConfiguration.fromPb(configurationPb);
        } else if (configurationPb.getLoad() != null) {
            return (T) LoadConfiguration.fromPb(configurationPb);
        } else if (configurationPb.getQuery() != null) {
            return (T) QueryJobConfiguration.fromPb(configurationPb);
        } else {
            throw new IllegalArgumentException("Job Configuration is not supported");
        }
    }
}
