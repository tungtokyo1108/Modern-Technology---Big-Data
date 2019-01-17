/**
 * Google - Big Data Technology
 * https://github.com/googleapis/google-cloud-java/blob/master/google-cloud-clients/google-cloud-bigquery/src/main/java/com/google/cloud/bigquery/BigQuery.java
 *
 *  Created on: Jan 17, 2019
 *  Student (MIG Virtual Developer): Tung Dang
 */

 package com.google.cloud.bigquery.mirror;

 import static com.google.common.base.Preconditions.checkArgument;

 import com.google.api.core.InternalApi;
 import com.google.api.gax.paging.Page;
 import com.google.cloud.FieldSelector;
 import com.google.cloud.FieldSelector.Helper;
 import com.google.cloud.RetryOption;
 import com.google.cloud.Service;
 import com.google.cloud.bigquery.spi.v2.BigQueryRpc;
 import com.google.common.base.Function;
 import com.google.common.collect.ImmutableList;
 import com.google.common.collect.Lists;

 import com.google.cloud.bigquery.BigQueryOptions;
 
 import java.io.Serializable;
 import java.util.ArrayList;
 import java.util.List;

 public interface BigQuery extends Service<BigQueryOptions> 
 {
     enum DatasetField implements FieldSelector 
     {
         ACCESS("access"),
         CREATION_TIME("creationTime"),
         DATASET_REFERENCE("datasetReference"),
         DEFAULT_TABLE_EXPIRATION_MS("defaultTableExpirationMsS"),
         DESCRIPTION("description"),
         ETAG("etag"),
         FRIENDLY_NAME("friendlyName"),
         ID("id"),
         LABELS("labels"),
         LAST_MODIFIED_TIME("lastModifiedTime"),
         LOCATION("location"),
         SELF_LINK("selfLink");

        static final List <? extends FieldSelector> REQUIRED_FIELDS = 
            ImmutableList.of(DATASET_REFERENCE);

        private final String selector;

        DatasetField(String selector) 
        {
            this.selector = selector;
        }

        @Override
        public String getSelector() 
        {
            return selector;
        }
     }

     enum TableField implements FieldSelector 
     {
        CREATION_TIME("creationTime"),
        DESCRIPTION("discription"),
        ETAG("etag"),
        EXPIRATION_TIME("expirationTime"),
        EXTERNAL_DATA_CONFIGURATION("externalDataConfiguration"),
        FRIENDLY_NAME("friendlyName"),
        ID("id"),
        LABELS("labels"),
        LAST_MODIFIED_TIME("lastModifiedTime"),
        LOCATION("location"),
        NUM_BYTES("numBytes"),
        NUM_ROWS("numRows"),
        SCHEMA("schema"),
        SELF_LINK("selfLink"),
        STREAMING_BUFFER("streamingBuffer"),
        TABLE_REFERENCE("tableReference"),
        TIME_PARTITIONING("timePartitioning"),
        TYPE("type"),
        VIEW("view");

        static final List <? extends FieldSelector> REQUIRED_FIELDS = 
            ImmutableList.of(TABLE_REFERENCE, TYPE);
        
        private final String selector;

        TableField(String selector)
        {
            this.selector = selector;
        }

        @Override
        public String getSelector()
        {
            return selector;
        }
     }
 }