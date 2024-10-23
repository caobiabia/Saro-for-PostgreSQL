select  count(*) from votes as v,          badges as b,         users as u where u.Id = v.UserId 	and v.UserId = b.UserId  AND u.Views<=64  AND u.DownVotes>=0  AND v.VoteTypeId=2;
