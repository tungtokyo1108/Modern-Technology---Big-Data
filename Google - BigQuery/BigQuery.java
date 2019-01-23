/**
 * Google - Big Data Technology
 * https://github.com/googleapis/google-cloud-java/blob/master/google-cloud-clients/google-cloud-bigquery/src/main/java/com/google/cloud/bigquery/BigQuery.java
 *
 *  Created on: Jan 17, 2019
 *  Student (MIG Virtual Developer): Tung Dang
 */

package com.google.cloud.bigquery.mirror;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.api.client.util.Data;
import com.google.api.core.InternalApi;
import com.google.api.gax.paging.Page;
import com.google.api.services.bigquery.Bigquery;
import com.google.api.services.bigquery.Bigquery.Jobs.Query;
import com.google.api.services.bigquery.model.Dataset;
import com.google.api.services.bigquery.model.DatasetList;
import com.google.api.services.bigquery.model.Job;
import com.google.api.services.bigquery.model.JobList;
import com.google.api.services.bigquery.model.JobStatus;
import com.google.api.services.bigquery.model.QueryResponse;
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
import com.google.cloud.bigquery.DatasetId;
import com.google.cloud.bigquery.DatasetInfo;
import com.google.cloud.bigquery.InsertAllRequest;
import com.google.cloud.bigquery.InsertAllResponse;
import com.google.cloud.bigquery.JobException;
import com.google.cloud.bigquery.JobId;
import com.google.cloud.bigquery.JobInfo;
import com.google.cloud.bigquery.QueryJobConfiguration;
import com.google.cloud.bigquery.Schema;
import com.google.cloud.bigquery.TableDataWriteChannel;
import com.google.cloud.bigquery.TableInfo;
import com.google.cloud.bigquery.TableResult;
import com.google.cloud.bigquery.WriteChannelConfiguration;

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

    class QueryResultsOption extends Option 
    {
        private static final long serialVersionUID = 3788898503226985525L;

        private QueryResultsOption(BigQueryRpc.Option option, Object value) 
        {
            super(option, value);
        }

        public static QueryResultsOption pageSize(long pageSize)
        {
            checkArgument(pageSize >= 0);
            return new QueryResultsOption(BigQueryRpc.Option.MAX_RESULTS, pageSize);
        }

        public static QueryResultsOption pageToken(String pageToken)
        {
            return new QueryResultsOption(BigQueryRpc.Option.PAGE_TOKEN, pageToken);
        }

        public static QueryResultsOption startIndex(long startIndex)
        {
            checkArgument(startIndex >= 0);
            return new QueryResultsOption(BigQueryRpc.Option.START_INDEX, startIndex);
        }

        public static QueryResultsOption maxWaitTime(long maxWaitTime)
        {
            checkArgument(maxWaitTime >= 0);
            return new QueryResultsOption(BigQueryRpc.Option.TIMEOUT, maxWaitTime);
        }
    }

    class QueryOption implements Serializable
    {
        private static final long serialVersionUID = 6206193419355824689L;

        private final Object option;

        private QueryOption(Object option)
        {
            this.option = option;
        }

        public QueryResultsOption getQueryResultsOption() 
        {
            return option instanceof QueryResultsOption ? (QueryResultsOption) option : null;
        }

        public RetryOption getRetryOption()
        {
            return option instanceof RetryOption ? (RetryOption) option : null;
        }

        static QueryResultsOption[] filterQueryResultsOptions(QueryOption... options)
        {
            List<QueryResultsOption> queryResultsOptions = new ArrayList<>(options.length);
            for (QueryOption opt : options)
            {
                if (opt.getQueryResultsOption() != null)
                {
                    queryResultsOptions.add(opt.getQueryResultsOption());
                }
            }
            return queryResultsOptions.toArray(new QueryResultsOption[queryResultsOptions.size()]);
        }

        static RetryOption[] filterRetryOptions(QueryOption... options)
        {
            List<RetryOption> retryOptions = new ArrayList<>(options.length);
            for (QueryOption opt : options)
            {
                if (opt.getRetryOption() != null)
                {
                    retryOptions.add(opt.getRetryOption());
                }
            }
            return retryOptions.toArray(new RetryOption[retryOptions.size()]);
        }

        public static QueryOption of(QueryResultsOption resultsOption)
        {
            return new QueryOption(resultsOption);
        }

        public static QueryOption of(RetryOption waitOption)
        {
            return new QueryOption(waitOption);
        }

        @Override 
        public int hashCode() 
        {
            return option != null ? option.hashCode() : 0;
        }

        @Override
        public boolean equals(Object obj) 
        {
            if (this == obj)
            {
                return true;
            }
            if (obj == null || getClass() != obj.getClass())
            {
                return false;
            }

            QueryOption that = (QueryOption) obj;
            return option != null ? option.equals(that.option) : that.option == null;
        }
    }

    Dataset create(DatasetInfo datasetInfo, DatasetOption...options);
    Table create(TableInfo tableInfo, TableOption... options);
    Job create(JobInfo jobInfo, JobOption... options);

    Dataset getDataset(String datasetId, DatasetOption... options);
    Dataset getDataset(DatasetId datasetId, DatasetOption... options);

    Page<Dataset> listDatasets(DatasetListOption... options);
    Page<Dataset> listDatasets(String projectId, DatasetListOption... options);
    Page<Table> listTables(String datasetId, TableListOption...options);
    Page<Table> listTables(DatasetId datasetId, TableListOption...options);
    
    boolean delete(String datasetId, DatasetDeleteOption...options);
    boolean delete(DatasetId datasetId, DatasetDeleteOption... options);
    boolean delete(String datasetId, String tableId);
    boolean delete(TableId tableId);

    Dataset update(DatasetInfo datasetInfo, DatasetOption...options);
    Table update(TableInfo tableInfo, TableOption...options);
    
    Table getTable(String datasetId, String tableId, TableOption...options);
    Table getTable(TableId tableId, TableOption... options);
    TableResult listTableData(String datasetId, String tableId, TableDataListOpt... options);
    TableResult listTableData(TableId tableId, TableDataListOpt... options);
    TableResult listTableData(String datasetId, String tableId, Schema schema, TableDataListOpt... options);
    TableResult listTableData(TableId tableId, Schema schema, TableDataListOpt... options);

    InsertAllResponse insretAll(InsertAllRequest request);

    Job getJob(String jobId, JobOption...options);
    Job getJob(JobId jobId, JobOption...options);
    Page<Job> listJobs(JobListOption...options);

    boolean cancel(String jobId);
    boolean cancel(JobId jobId);

    TableResult query(QueryJobConfiguration configuration, JobOption...options)
        throws InterruptedException, JobException;
    
    TableResult query(QueryJobConfiguration configuration, JobId jobId, JobOption...options)
        throws InterruptedException, JobException;

    @InternalApi
    QueryResponse getQueryResponse(JobId jobId, QueryResultsOption...options);

    TableDataWriteChannel writer(WriteChannelConfiguration writeChannelConfiguration);
    TableDataWriteChannel writer(JobId jobId, WriteChannelConfiguration writeChannelConfiguration);
}

