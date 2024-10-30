select  count(*) from votes as v,          badges as b,         users as u where u.Id = v.UserId 	and v.UserId = b.UserId  AND u.UpVotes>=0  AND u.UpVotes<=17;
