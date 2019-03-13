/**
 * Google - Big Data Technology
 * https://github.com/googleapis/google-cloud-java/blob/master/google-cloud-clients/google-cloud-bigquery/src/main/java/com/google/cloud/bigquery/BigQueryOptions.java
 *
 *  Created on: Mar 13, 2019
 *  Data Scientist: Tung Dang
 */

package com.google.cloud.bigquery.mirror;

import com.google.cloud.ServiceDefaults;
import com.google.cloud.ServiceOptions;
import com.google.cloud.ServiceRpc;
import com.google.cloud.TransportOptions;
import com.google.cloud.bigquery.spi.BigQueryRpcFactory;
import com.google.cloud.bigquery.BigQueryFactory;
import com.google.cloud.bigquery.mirror.BigQueryRpc;
import com.google.cloud.bigquery.mirror.HttpBigQueryRpc;
import com.google.cloud.http.HttpTransportOptions;
import com.google.common.collect.ImmutableSet;
import com.google.errorprone.annotations.Immutable;

import java.util.Set;

public class BigQueryOptions extends ServiceOptions<BigQuery, BigQueryOptions>
{
    private static final String API_SHORT_NAME = "BigQuery";
    private static final String BIGQUERY_SCOPE = "https://www.googleapis.com/auth/bigquery";
    private static final Set<String> SCOPES = ImmutableSet.of(BIGQUERY_SCOPE);
    private static final long serialVersionUID = -2437598817433266049L;
    private final String location;
    private boolean setThrowNotFound;

    public static class DefaultBigQueryFactory implements BigQueryFactory
    {
        private static final BigQueryFactory INSTANCE = new DefaultBigQueryFactory();
        @Override
        public BigQuery create(BigQueryOptions options)
        {
            return BigQueryImpl(options);
        }
    }

    public static class DefaultBigQueryRpcFactory implements BigQueryRpcFactory
    {
        private static final BigQueryRpcFactory INSTANCE = new DefaultBigQueryRpcFactory();
        @Override
        public ServiceRpc create(BigQueryOptions options)
        {
            return new HttpBigQueryRpc(options);
        }
    }

    public static class Builder extends ServiceOptions.Builder<BigQuery, BigQueryOptions, Builder>
    {
        private String location;
        private Builder() {}
        private Builder(BigQueryOptions options)
        {
            super(options);
        }

        @Override
        public Builder setTransportOptions(TransportOptions transportOptions)
        {
            if (!(transportOptions instanceof HttpTransportOptions))
            {
                throw new IllegalArgumentException(
                    "Only http transport is allowed for" + API_SHORT_NAME + ".");
            }
            return super.setTransportOptions(transportOptions);
        }

        public Builder setLocation(String location)
        {
            this.location = location;
            return this;
        }

        @Override
        public BigQueryOptions build()
        {
            return new BigQueryOptions(this);
        }
    }

    private BigQueryOptions(Builder builder)
    {
        super(BigQueryFactory.class, BigQueryRpcFactory.class, builder, new BigQueryDefaults());
        this.location = builder.location;
    }

    private static class BigQueryDefaults implements ServiceDefaults<BigQuery, BigQueryOptions>
    {
        @Override 
        public BigQueryFactory getDefaultServiceFactory()
        {
            return DefaultBigQueryFactory.INSTANCE;
        }

        @Override
        public BigQueryRpcFactory getDefaultRpcFactory()
        {
            return DefaultBigQueryRpcFactory.INSTANCE;
        }

        @Override
        public TransportOptions getDefaulTransportOptions()
        {
            return getDefaulTransportOptions();
        }
    }

    public static HttpTransportOptions getDefaultHttpTransportOptions()
    {
        return HttpTransportOptions.newBuilder().build();
    }

    @Override
    protected Set<String> getScopes()
    {
        return SCOPES;
    }

    protected BigQueryRpc getBigQueryRpcV2()
    {
        return (BigQueryRpc) getRpc();
    }

    public String getLocation()
    {
        return location;
    }

    public void setThrowNotFound(boolean setThrowNotFound)
    {
        this.setThrowNotFound = setThrowNotFound;
    }

    public boolean getThrowNotFound()
    {
        return setThrowNotFound;
    }

    @SuppressWarnings("unchecked")
    @Override
    public Builder toBuilder()
    {
        return new Builder(this);
    }

    @Override
    public int hashCode()
    {
        return baseHashCode();
    }

    @Override
    public boolean equals(Object obj)
    {
        if (!(obj instanceof BigQueryOptions))
        {
            return false;
        }
        BigQueryOptions other = (BigQueryOptions) obj;
        return baseEquals(other);
    }

    public static BigQueryOptions getDefaultInstance()
    {
        return newBuilder().build();
    }

    public static Builder newBuilder()
    {
        return new Builder();
    }
}
