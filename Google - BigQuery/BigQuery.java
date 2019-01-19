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
import com.google.api.services.bigquery.model.Dataset;
import com.google.api.services.bigquery.model.JobList;
import com.google.api.services.bigquery.model.JobStatus;
import com.google.api.services.bigquery.model.Table;
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

import javax.print.attribute.standard.RequestingUserName;
import javax.swing.text.html.Option;

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

    enum JobField implements FieldSelector 
    {
        CONFIGURATION("configuration"),
        ETAG("etag"),
        ID("id"),
        JOB_REFERENCE("jobReference"),
        SELF_LINK("selfLink"),
        STATISTICS("statistics"),
        STATUS("status"),
        USER_EMAIL("user_email");

        static final List<? extends FieldSelector> REQUIRED_FIELDS = 
            ImmutableList.of(JOB_REFERENCE, CONFIGURATION);
        
        private final String selector;

        JobField(String selector)
        {
            this.selector = selector;
        }

        @Override
        public String getSelector() 
        {
            return selector;
        }
    }

    class DatasetListOption extends Option 
    {
        private static final long serialVersionUID = 8660294969063340498L;

        private DatasetListOption(BigQueryRpc.Option option, Object value) 
        {
            super(option, value);
        }

        public static DatasetListOption pageSize(long pageSize)
        {
            return new DatasetListOption(BigQueryRpc.Option.MAX_RESULTS, pageSize);
        }

        public static DatasetListOption pageToken(String pageToken) 
        {
            return new DatasetListOption(BigQueryRpc.Option.PAGE_TOKEN, pageToken);
        }

        public static DatasetListOption all()
        {
            return new DatasetListOption(BigQueryRpc.Option.ALL_DATASETS, true);
        }
    }

    /**
     * Class for specifying data get, create and update options
     */
    class DatasetOption extends Option
    {
        private static final long serialVersionUID = 1674133909259913250L;

        private DatasetOption(BigQueryRpc.Option option, Object value) 
        {
            super(option, value);
        }

        public static DatasetOption fields(DatasetField... fields)
        {
            return new DatasetOption(
                BigQueryRpc.Option.FIELDS, Helper.selector(DatasetField.REQUIRED_FIELDS, fields)
            );
        }
    }

    /**
     * Class for specifying data get, delete options
     */
    class DatasetDeleteOption extends Option
    {
        private static final long serialVersionUID = -7166083569900951337L;

        private DatasetDeleteOption(BigQueryRpc.Option option, Object value) 
        {
            super(option, value);
        }

        public static DatasetDeleteOption deleteContents()
        {
            return new DatasetDeleteOption(BigQueryRpc.Option.DELETE_CONTENTS, true);
        }
    }

    class TableListOption extends Option 
    {
        private static final long serialVersionUID = 8660294969063340498L;

        private TableListOption(BigQueryRpc.Option option, Object value)
        {
            super(option, value);
        }

        public static TableListOption pageSize(long pageSize) 
        {
            checkArgument(pageSize >= 0);
            return new TableListOption(BigQueryRpc.Option.MAX_RESULTS, pageSize);
        }

        public static TableListOption pageToken(String pageToken)
        {
            return new TableListOption(BigQueryRpc.Option.PAGE_TOKEN, pageToken);
        }
    }

    class TableOption extends Option 
    {
        private static final long serialVersionUID = -1723870134095936772L;

        private TableOption(BigQueryRpc.Option option, Object value)
        {
            super(option, value);
        }

        public static TableOption fields(TableField...fields)
        {
            return new TableOption(
                BigQueryRpc.Option.FIELDS, Helper.selector(TableField.REQUIRED_FIELDS, fields)
            );
        }
    }

    class TableDataListOpt extends Option
    {
        private static final long serialVersionUID = 8488823381738864434L;

        private TableDataListOpt(BigQueryRpc.Option option, Object value)
        {
            super(option, value);
        }

        public static TableDataListOpt pageSize(long pageSize)
        {
            checkArgument(pageSize >= 0);
            return new TableDataListOpt(BigQueryRpc.Option.MAX_RESULTS, pageSize);
        }

        public static TableDataListOpt pageToken(String pageToken)
        {
            return new TableDataListOpt(BigQueryRpc.Option.PAGE_TOKEN, pageToken);
        }

        /**
         * Sets the zero-based index of the row from which to listing table
         */
        public static TableDataListOpt startIndex(long index)
        {
            checkArgument(index >= 0);
            return new TableDataListOpt(BigQueryRpc.Option.START_INDEX, index);
        }
    }

    class JobListOption extends Option
    {
        private static final long serialVersionUID = -8207122131226481423L;

        private JobListOption(BigQueryRpc.Option option, Object value)
        {
            super(option, value);
        }

        public static JobListOption allUsers() 
        {
            return new JobListOption(BigQueryRpc.Option.ALL_USERS, true);
        }

        public static JobListOption stateFilter(JobStatus.State... stateFilters)
        {
            List<String> stringFilters = 
                Lists.transform(
                    ImmutableList.copyOf(stateFilters),
                    new Function<JobStatus.State, String>() {
                        @Override
                        public String apply(JobStatus.State state) 
                        {
                            return state.name().toLowerCase();
                        }
                    }
                );
            return new JobListOption(BigQueryRpc.Option.STATE_FILTER, stringFilters);
        }

        public static JobListOption pageSize(long pageSize)
        {
            checkArgument(pageSize >= 0);
            return new JobListOption(BigQueryRpc.Option.MAX_RESULTS, pageSize);
        }

        public static JobListOption pageToken(String pageToken)
        {
            return new JobListOption(BigQueryRpc.Option.PAGE_TOKEN, pageToken);
        }

        public static JobListOption fields(JobField...fields)
        {
            return new JobListOption(
                BigQueryRpc.Option.FIELDS,
                Helper.listSelector("jobs", JobField.REQUIRED_FIELDS, fields, "state", "errorResult")
            );
        }
    }

    class JobOption extends Option
    {
        private static final long serialVersionUID = -3111736712316353665L;

        private JobOption(BigQueryRpc.Option option, Object value)
        {
            super(option, value);
        }

        public static JobOption fields(JobField...fields)
        {
            return new JobOption(
                BigQueryRpc.Option.FIELDS, Helper.selector(JobField.REQUIRED_FIELDS, fields)
            );
        }
    }
 }
