/**
 * Google - Big Data Technology
 * https://github.com/googleapis/google-cloud-java/blob/master/google-cloud-clients/google-cloud-bigquery/src/main/java/com/google/cloud/bigquery/JobStatus.java
 *  Created on: Jan 18, 2019
 *  Student (MIG Virtual Developer): Tung Dang
 */

package com.google.cloud.bigquery.mirror;

import com.google.api.core.ApiFunction;
import com.google.cloud.StringEnumType;
import com.google.cloud.StringEnumValue;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import java.io.Serializable;
import java.util.List;
import java.util.Objects;
import javax.annotation.Nullable;

import com.google.cloud.bigquery.BigQueryError;

public class JobStatus implements Serializable 
{
    private static final long serialVersionUID = -714976456815445365L;

    public static final class State extends StringEnumType
    {
        private static final long serialVersionUID = 818920627219751204L;

        private static final ApiFunction<String, State> CONSTRUCTOR = 
            new ApiFunction<String,State>() {
                @Override
                public State apply(String constant)
                {
                    return new State(constant);
                }
            };

        private static final StringEnumType<State> type = new StringEnumType(State.class, CONSTRUCTOR);

        public static final State PENDING = type.createAndRegister("PENDING");
        public static final State RUNNING = type.createAndRegister("RUNNING");
        public static final State DONE = type.createAndRegister("DONE");

        private State (String constant)
        {
            super(constant);
        }

        public static State valueOfStrict(String constant)
        {
            return type.valueOfStrict(constant);
        }

        public static State valueOf(String constant)
        {
            return type.valueOf(constant);
        }

        public static State[] values() 
        {
            return type.values();
        }
    }

    private final State state;
    private final BigQueryError error;
    private final List<BigQueryError> executionErrors;

    JobStatus(State state) 
    {
        this.state = state;
        this.error = null;
        this.executionErrors = null;
    }

    JobStatus(State state, BigQueryError error, List<BigQueryError> executionErrors)
    {
        this.state = state;
        this.error = error;
        this.executionErrors = executionErrors;
    }

    public State getState() 
    {
        return state;
    }

    @Nullable
    public BigQueryError getError()
    {
        return error;
    }

    public List<BigQueryError> getExecutionErrors()
    {
        return executionErrors;
    }

    @Override
    public String toString()
    {
        return MoreObjects.toStringHelper(this)
            .add("state", state)
            .add("error", error)
            .add("executionErrors", executionErrors)
            .toString();
    }

    @Override
    public final int hashCode() 
    {
        return Objects.hash(state, error, executionErrors);
    }

    @Override
    public final boolean equals(Object obj)
    {
        return obj == this
            || obj != null
                && obj.getClass().equals(JobStatus.class)
                && Objects.equals(toPb(), ((JobStatus) obj).toPb());
    }

    com.google.api.services.bigquery.model.JobStatus toPb() 
    {
        com.google.api.services.bigquery.model.JobStatus statusPb = 
            new com.google.api.services.bigquery.model.JobStatus();
        if (state != null)
        {
            statusPb.setState(state.toString());
        }
        if (error != null)
        {
            statusPb.setErrorResult(error.toPb());
        }
        if (executionErrors != null)
        {
            statusPb.setErrors(Lists.transform(executionErrors, BigQueryError.TO_PB_FUNCTION));
        }
        return statusPb;
    }

    static JobStatus fromPb(com.google.api.services.bigquery.model.JobStatus statusPb) 
    {
        List<BigQueryError> allErrors = null;
        if (statusPb.getErrors() != null)
        {
            allErrors = List.transform(statusPb.getErrors(), BigQueryError.FROM_PB_FUNCTION);
        }
        BigQueryError error = statusPb.getErrorResult() != null ? BigQueryError.fromPb(statusPb, getErrorResult()) : null;
        return new JobStatus(State.valueOf(statusPb.getState()), error, allErrors);
    }
}
