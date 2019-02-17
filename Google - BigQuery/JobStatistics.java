/**
 * Google - Big Data Technology
 * https://github.com/googleapis/google-cloud-java/blob/master/google-cloud-clients/google-cloud-bigquery/src/main/java/com/google/cloud/bigquery/JobStatistics.java
 *
 *  Created on: Feb 17, 2019
 *  Student (MIG Virtual Developer): Tung Dang
 */

package com.google.cloud.bigquery.mirror;

import com.google.api.core.ApiFunction;
import com.google.api.services.bigquery.model.JobConfiguration;
import com.google.api.services.bigquery.model.JobStatistics2;
import com.google.api.services.bigquery.model.JobStatistics3;
import com.google.api.services.bigquery.model.JobStatistics4;
import com.google.cloud.StringEnumType;
import com.google.cloud.StringEnumValue;
import com.google.common.base.MoreObjects;
import com.google.common.base.MoreObjects.ToStringHelper;
import com.google.common.collect.Lists;
import com.google.errorprone.annotations.OverridingMethodsMustInvokeSuper;
import com.google.rpc.RetryInfo;

import java.io.Serializable;
import java.util.List;
import java.util.Objects;

public abstract class JobStatistics implements Serializable 
{
    private static final long serialVersionUID = 1433024714741660399L;

    private final Long creationTime;
    private final Long endTime;
    private final Long startTime;

    public static class CopyStatistics extends JobStatistics
    {
        private static final long serialVersionUID = 8218325588441660938L;

        static final class Builder extends JobStatistics.Builder<CopyStatistics, Builder>
        {
            private Builder() {}
            private Builder(com.google.api.services.bigquery.model.JobStatistics statisticsPb)
            {
                super(statisticsPb);
            }
            @Override
            CopyStatistics build()
            {
                return new CopyStatistics(this);
            }
        }

        private CopyStatistics(Builder builder)
        {
            super(builder);
        }

        @Override
        public final boolean equals(Object obj) {
            return obj == this
                || obj != null
                    && obj.getClass().equals(CopyStatistics.class)
                    && baseEquals((CopyStatistics) obj);
        }

        @Override
        public final int hashCode()
        {
            return baseHashCode();
        }

        static Builder newBuilder()
        {
            return new Builder();
        }

        @SuppressWarnings("unchecked")
        static CopyStatistics fromPb(com.google.api.services.bigquery.model.JobStatistics statisticsPb)
        {
            return new Builder(statisticsPb).build();
        }
    }

    public static class ExtractStatistics extends JobStatistics 
    {
        private static final long serialVersionUID = -1566598819212767373L;

        private final List<Long> destinationUriFileCounts;
        static final class Builder extends JobStatistics.Builder<ExtractStatistics, Builder>
        {
            private List<Long> destinationUriFileCounts;
            private Builder() {}
            private Builder(com.google.api.services.bigquery.model.JobStatistics statisticsPb)
            {
                super(statisticsPb);
                if (statisticsPb.getExtract() != null)
                {
                    this.destinationUriFileCounts = statisticsPb.getExtract().getDestinationUriFileCounts();
                }
            }

            Builder setDestinationFileCounts(List<Long> destinationUriFileCounts)
            {
                this.destinationUriFileCounts = destinationUriFileCounts;
                return self();
            }

            @Override
            ExtractStatistics build()
            {
                return new ExtractStatistics(this);
            }
        }

        private ExtractStatistics(Builder builder)
        {
            super(builder);
            this.destinationUriFileCounts = builder.destinationUriFileCounts;
        }

        public List<Long> getDestinationUriFileCounts()
        {
            return destinationUriFileCounts;
        }

        @Override
        ToStringHelper toStringHelper() 
        {
            return super.toStringHelper().add("destinationUriFileCounts", destinationUriFileCounts);
        }

        @Override
        public final boolean equals(Object obj)
        {
            return obj == this
                || obj != null
                    && obj.getClass().equals(ExtractStatistics.class)
                    && baseEquals((ExtractStatistics) obj);
        }

        @Override
        public final int hashCode()
        {
            return Objects.hash(baseHashCode(), destinationUriFileCounts);
        }

        @Override
        com.google.cloud.bigquery.model.JobStatistics toPb()
        {
            com.google.cloud.bigquery.model.JobStatistics statisticsPb = super.toPb();
            return statisticsPb.setExtract(
                new JobStatistics4().setDestinationUriFileCounts(destinationUriFileCounts));
        }

        static Builder newBuilder()
        {
            return new Builder();
        }

        @SuppressWarnings("unchecked")
        static ExtractStatistics fromPb(
            com.google.api.services.bigquery.model.JobStatistics statisticsPb) {
                return new Builder(statisticsPb).build();
        }
    }


}
